"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.special import roots_genlaguerre, roots_laguerre
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 620.0  # safety margin before 630s kill

def _objective(r):
    """
    Computes the C4 upper bound proxy:
    Integral of |P(x)|^2 * x * exp(-x) dx over [0, inf).
    P(x) = prod (x - r_i)^2
    """
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0) or np.any(np.isnan(r)):
        return 1e12
    
    # Use high-order Gauss-Laguerre quadrature for weight x^1 * exp(-x)
    # alpha=1 for weight x^alpha * exp(-x)
    # We need enough nodes to integrate a polynomial of degree 2*12=24 accurately.
    # Gauss-Laguerre with N nodes is exact for polynomials up to degree 2N-1.
    # So N=25 is exact for degree 49. Let's use N=50 for safety and numerical stability.
    try:
        nodes, weights = roots_genlaguerre(50, 1)
    except Exception:
        # Fallback to standard Laguerre if generalized fails (unlikely)
        from numpy.polynomial.laguerre import laggauss
        nodes, w_std = laggauss(50)
        weights = w_std * nodes # Approximation for x*exp(-x) if generalized not available
        
    # Compute P(x) at nodes in log-space to prevent overflow
    log_abs_P = np.zeros_like(nodes)
    for ri in r:
        diff = nodes - ri
        # Add small epsilon to avoid log(0) if node == root
        log_abs_P += 2.0 * np.log(np.abs(diff) + 1e-300)
    
    # Integrand is |P(x)|^2 * x * exp(-x)
    # In quadrature: sum(weights * |P(x_i)|^2)
    # log(integrand_val) = 2 * log_abs_P + log(weight)
    log_val = 2.0 * log_abs_P + np.log(weights + 1e-300)
    
    if np.any(np.isnan(log_val)):
        return 1e12
        
    max_log_val = np.max(log_val)
    if max_log_val > 700: # Overflow protection for exp
        # This shouldn't happen if roots are reasonable, but just in case
        return 1e12
        
    integral = np.sum(np.exp(log_val - max_log_val)) * np.exp(max_log_val)
    
    return integral

def _run_optimization(initial_r, max_iter=2000):
    """
    Runs L-BFGS-B optimization with time guard.
    """
    def obj(x):
        if time.time() - _START_TIME > _TIME_LIMIT:
            raise TimeoutError()
        return _objective(x)
    
    try:
        result = minimize(
            obj,
            initial_r,
            method='L-BFGS-B',
            bounds=[(0.01, None)] * 12,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return result.x, result.fun
    except TimeoutError:
        return initial_r, _objective(initial_r)
    except Exception:
        return initial_r, _objective(initial_r)

def construct():
    """
    Returns list of 12 positive double-root locations for the Laguerre
    combination (m=12, n=1) that minimizes the C4 upper bound.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # 1. Generate candidate starting points
    
    candidates = []
    
    # A. Seed: Arithmetic progression
    seed_r = np.array([4.0 * (i + 1) for i in range(12)])
    candidates.append(seed_r)
    
    # B. Zeros of Generalized Laguerre L_12^(1)(x)
    # This is the natural orthogonal polynomial for weight x^1 e^-x
    try:
        zeros_gen, _ = roots_genlaguerre(12, 1)
        candidates.append(zeros_gen)
        
        # C. Scaled Generalized Laguerre Zeros
        # The seed has mean ~26. The zeros of L_12^(1) have mean ~ 2*12 + 1 = 25 (approx)
        # Actually mean of L_n^(alpha) zeros is n + (alpha+1)/2. For n=12, alpha=1, mean=12.5.
        # The seed mean is 26. So we might need to scale by factor ~2.
        # Let's try scaling to match the mean of the seed.
        mean_seed = np.mean(seed_r)
        mean_zeros = np.mean(zeros_gen)
        if mean_zeros > 0:
            scale = mean_seed / mean_zeros
            scaled_zeros = zeros_gen * scale
            candidates.append(scaled_zeros)
            
            # Also try scaling to match the range
            range_seed = np.max(seed_r) - np.min(seed_r)
            range_zeros = np.max(zeros_gen) - np.min(zeros_gen)
            if range_zeros > 0:
                scale_range = range_seed / range_zeros
                # Shift to match min
                shifted_scaled = (zeros_gen - np.min(zeros_gen)) * scale_range + np.min(seed_r)
                candidates.append(shifted_scaled)
    except Exception:
        pass
        
    # D. Zeros of Standard Laguerre L_12(x) (alpha=0)
    try:
        zeros_std, _ = roots_laguerre(12)
        candidates.append(zeros_std)
        
        # E. Scaled Standard Laguerre Zeros
        mean_std = np.mean(zeros_std)
        if mean_std > 0:
            scale_std = mean_seed / mean_std
            candidates.append(zeros_std * scale_std)
    except Exception:
        pass
        
    # F. Random restarts (few, for safety)
    np.random.seed(42)
    for _ in range(3):
        r_rand = np.sort(np.random.uniform(1.0, 50.0, 12))
        candidates.append(r_rand)

    # 2. Evaluate all candidates without optimization first
    best_r = None
    best_val = np.inf
    
    for r in candidates:
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        val = _objective(r)
        if val < best_val:
            best_val = val
            best_r = r.copy()
            
    # 3. Run optimization from the best candidates
    # Sort candidates by their initial objective value
    cand_scores = [(r, _objective(r)) for r in candidates]
    cand_scores.sort(key=lambda x: x[1])
    
    # Take top 3 distinct candidates for optimization
    optimized_results = []
    for r_init, _ in cand_scores[:3]:
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
            
        # Estimate time remaining
        elapsed = time.time() - _START_TIME
        remaining = _TIME_LIMIT - elapsed - 30
        if remaining < 5:
            break
            
        # Dynamic max_iter based on remaining time
        # Assume ~1ms per iteration for this size
        max_iter = int(min(2000, max(200, remaining * 500)))
        
        r_opt, val_opt = _run_optimization(r_init, max_iter=max_iter)
        optimized_results.append((r_opt, val_opt))
        
        if val_opt < best_val:
            best_val = val_opt
            best_r = r_opt.copy()

    # 4. Final check and return
    if best_r is None:
        return [4.0 * (i + 1) for i in range(12)]
        
    # Ensure positivity and sorting
    best_r = np.maximum(best_r, 0.01)
    best_r = np.sort(best_r)
    
    return [float(x) for x in best_r]

# EVOLVE-BLOCK-END
