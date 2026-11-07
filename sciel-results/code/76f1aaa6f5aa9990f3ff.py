"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 135.0  # safety margin below 150s kill
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def objective(h):
    """Compute the raw value (lower is better)."""
    if not np.all(np.isfinite(h)):
        return float('inf')
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def project(h):
    """Project h to satisfy: h in [0,1]^N, sum(h) ~ N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    """
    Strategy: Cyclic Symmetry Reduction + Convex QP Solving.
    
    We assume the optimal solution has a periodic structure with period p,
    where p divides N. We search over all divisors p of N.
    For a fixed period p, we define h as a repetition of a base vector x of length p.
    The constraints become:
    1. sum(x) = p/2 (since total sum is N/2 and there are N/p repeats)
    2. 0 <= x_i <= 1
    3. Maximize min_k (autocorrelation of x at lag k mod p)
    
    Actually, we want to minimize max_k correlate(h, 1-h)[k].
    For periodic h, correlate(h, 1-h)[k] depends on k mod p.
    Let A_k = sum_{i=0}^{p-1} x_i (1 - x_{(i+k)%p}).
    We want to minimize max_k A_k.
    
    This is a convex optimization problem for fixed p because:
    A_k(x) is linear in x? No, A_k = sum x_i - sum x_i x_{i+k}.
    The term -sum x_i x_{i+k} is concave.
    So A_k is concave.
    Minimizing a concave function is non-convex.
    However, we can use SLSQP which handles non-convex problems locally well,
    especially in low dimensions (p <= 100).
    
    We will iterate p over divisors of N.
    For each p, we run a local optimization (SLSQP) starting from a good initial guess
    (e.g., alternating 0s and 1s, or random).
    """
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # 1.0
    
    # Divisors of N
    divisors = [d for d in range(1, N + 1) if N % d == 0]
    
    # Sort divisors to try smaller periods first (often better structure)
    # But also try p=1, p=2, etc.
    # p=1: all h_i = 0.5. value = 1.0.
    # p=2: [a, b, a, b...]. sum = 50(a+b)=50 => a+b=1.
    # Let's just run the optimizer for each divisor.
    
    for p in divisors:
        if remaining_time() < 10:
            break
            
        # Initial guess: Alternating 0.5? Or random?
        # A good guess for minimizing overlap is to spread out the mass.
        # For p, we can try a "balanced" vector.
        # Let's use a few random starts for robustness.
        
        starts = []
        # Start 1: Uniform
        starts.append(np.full(p, 0.5))
        # Start 2: Random
        for _ in range(3):
            if remaining_time() < 5:
                break
            x_rand = np.random.uniform(0, 1, p)
            # Project to sum p/2
            x_rand = x_rand - x_rand.mean() + 0.5
            x_rand = np.clip(x_rand, 0, 1)
            # Adjust sum
            diff = p/2 - x_rand.sum()
            x_rand += diff / p
            x_rand = np.clip(x_rand, 0, 1)
            starts.append(x_rand)
        
        for x0 in starts:
            if remaining_time() < 5:
                break
                
            def obj_and_grad(x):
                # x is the base vector of length p
                # Construct h
                h = np.tile(x, N // p)
                
                # Compute correlations for lags 0 to p-1
                # Note: correlate(h, 1-h, 'full') has length 2N-1.
                # For periodic h, c_k depends only on k % p.
                # We only need to check k = 0, ..., p-1.
                # c_k = sum_{i=0}^{N-1-k} h_i (1-h_{i+k})
                # Since h is periodic, this sum is (N//p) * sum_{i=0}^{p-1} x_i (1-x_{(i+k)%p})
                # Wait, the sum in correlate is over the overlap.
                # For lag k, the number of terms is N-k.
                # This breaks the simple periodicity scaling if N-k is not a multiple of p.
                # However, for large N and small p, the max is likely dominated by the periodic part.
                # Let's just compute the full correlation for the constructed h.
                
                c = np.correlate(h, 1.0 - h, mode="full")
                val = np.max(c) / N * 2.0
                
                # Gradient is hard to compute analytically for the max.
                # SLSQP will use numerical gradients or we can provide approximations.
                # Given the small dimension (p), numerical gradients are fine.
                return val
            
            # Constraints
            # Sum(x) = p/2
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - p / 2.0}
            ]
            bounds = [(0.0, 1.0)] * p
            
            try:
                res = minimize(
                    obj_and_grad,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 50, 'ftol': 1e-8}
                )
                
                if res.success or res.fun < best_val:
                    x_opt = res.x
                    # Project to ensure feasibility
                    x_opt = np.clip(x_opt, 0, 1)
                    diff = p/2 - x_opt.sum()
                    x_opt += diff / p
                    x_opt = np.clip(x_opt, 0, 1)
                    
                    h_candidate = np.tile(x_opt, N // p)
                    val_candidate = objective(h_candidate)
                    
                    if val_candidate < best_val:
                        best_val = val_candidate
                        best_h = h_candidate.copy()
                        
            except Exception:
                pass
                
    # Final local refinement on the best h found using Nelder-Mead on the full vector
    # This can break the periodic symmetry if beneficial.
    if remaining_time() > 15:
        def obj_full(h_flat):
            h = project(h_flat)
            return objective(h)
        
        res = minimize(
            obj_full,
            best_h,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        h_res = project(res.x)
        val_res = objective(h_res)
        
        if val_res < best_val:
            best_val = val_res
            best_h = h_res.copy()

    # Final sanity check
    best_h = project(best_h)
    assert np.all(np.isfinite(best_h)), "Non-finite values in result"
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Out of bounds"
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h


# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
