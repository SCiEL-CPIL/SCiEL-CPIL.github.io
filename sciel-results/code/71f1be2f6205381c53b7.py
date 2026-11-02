import numpy as np
import time
from scipy.optimize import minimize, linprog

def construct():
    """
    Find step heights h on [0, 2] (with W = 2/N, sum(h) = N/2) to minimize
    the maximum overlap of h with shifted (1-h).

    The overlap at lag k is:
        overlap[k] = sum_i h_i * (1 - h_{i+k mod N})
                   = sum_i h_i - sum_i h_i * h_{i+k mod N}
                   = N/2 - autocorr_k(h)
    where autocorr_k(h) = sum_i h_i * h_{i+k mod N}.

    Minimizing max_k overlap[k] is equivalent to maximizing min_k autocorr_k(h)
    subject to h in [0, 1], sum(h) = N/2.

    This is a convex optimization problem (minimize t s.t. t >= max_k overlap[k],
    h in box, affine constraint). We use scipy.optimize.minimize with L-BFGS-B.
    """
    N = 100
    W = 2.0 / N
    target_sum = N / 2.0
    
    deadline = time.time() + 145  # Safety margin from 150s

    def _overlap_max(h):
        """Compute max overlap of h with shifted 1-h."""
        # Compute autocorrelation via FFT
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(H * np.conj(H)).real
        # autocorr[k] = sum_i h_i * h_{(i+k) mod N}
        # overlap[k] = target_sum - autocorr[k]
        overlaps = target_sum - autocorr
        return np.max(overlaps)
    
    def _soft_max_overlap(h, gamma=50.0):
        """Smooth max of overlaps using log-sum-exp."""
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(H * np.conj(H)).real
        overlaps = target_sum - autocorr
        # Log-sum-exp for numerical stability
        max_val = np.max(overlaps)
        if max_val > 1e10:
            return max_val
        log_sum_exp = max_val + np.log(np.sum(np.exp(overlaps - max_val)))
        return log_sum_exp / gamma
    
    def _soft_max_overlap_grad(h, gamma=50.0):
        """Gradient of smooth max overlap with respect to h."""
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(H * np.conj(H)).real
        overlaps = target_sum - autocorr
        
        max_val = np.max(overlaps)
        exp_vals = np.exp(overlaps - max_val)
        weights = exp_vals / np.sum(exp_vals)
        
        # grad overlap[k] w.r.t. h
        # overlap[k] = target_sum - autocorr[k]
        # autocorr[k] = sum_i h_i * h_{i+k}
        # d(autocorr[k])/dh_j = h_{j+k} + h_{j-k} (for circular)
        # d(overlap[k])/dh_j = -(h_{j+k} + h_{j-k})
        
        # grad_soft_max = (1/gamma) * sum_k weights[k] * grad overlap[k]
        # = (1/gamma) * sum_k weights[k] * (-(h_{j+k} + h_{j-k}))
        
        grad = np.zeros(N)
        for k in range(N):
            wk = weights[k]
            if wk < 1e-10:
                continue
            # h_{j+k mod N} for all j
            shifted_plus = np.roll(h, -k)
            shifted_minus = np.roll(h, k)
            grad -= wk * (shifted_plus + shifted_minus)
        
        grad /= gamma
        return grad
    
    def _objective_and_grad(h, gamma=50.0):
        obj = _soft_max_overlap(h, gamma)
        grad = _soft_max_overlap_grad(h, gamma)
        return obj, grad
    
    def _project_simplex_box(h, target):
        """Project h onto {x : 0 <= x_i <= 1, sum(x) = target}."""
        # Use bisection on Lagrange multiplier
        lo = 0.0
        hi = 1.0
        
        def sum_projected(mu):
            proj = np.clip(h - mu, 0.0, 1.0)
            return np.sum(proj)
        
        # Binary search for mu such that sum_projected(mu) = target
        for _ in range(100):
            mid = (lo + hi) / 2.0
            s = sum_projected(mid)
            if s > target:
                lo = mid
            else:
                hi = mid
        
        mu = (lo + hi) / 2.0
        return np.clip(h - mu, 0.0, 1.0)
    
    def _objective_constrained(h, gamma=50.0):
        """Objective with constraints enforced via penalty."""
        # First project to feasible set
        h_feasible = _project_simplex_box(h, target_sum)
        obj, _ = _objective_and_grad(h_feasible, gamma)
        return obj
    
    def _objective_constrained_grad(h, gamma=50.0):
        """Gradient for constrained objective (approximate via projection)."""
        h_feasible = _project_simplex_box(h, target_sum)
        _, grad = _objective_and_grad(h_feasible, gamma)
        return grad
    
    # Initial guess: uniform (score 1.0)
    h0 = np.full(N, 0.5)
    
    # Strategy: Use L-BFGS-B with soft max objective
    # We work in the space where sum(h) = target_sum is handled by projection
    # But L-BFGS-B works best with explicit constraints. Let's use a different approach.
    
    # Alternative: Parametrize h as 0.5 + z, where sum(z) = 0, and -0.5 <= z_i <= 0.5
    # Then use optimization in the (N-1)-dimensional subspace.
    
    # Let's use scipy.optimize.minimize with bounds and equality constraint
    # Variable: h of length N
    # Constraints: sum(h) = target_sum
    # Bounds: 0 <= h_i <= 1
    
    start_time = time.time()
    
    # Try multiple initializations with different random seeds
    best_h = h0.copy()
    best_score = _overlap_max(best_h)
    
    np.random.seed(42)
    
    # Multi-start: try a few random initial points
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        
        # Random initial h in [0, 1] with sum = target_sum
        if trial == 0:
            h_init = h0.copy()
        else:
            # Random perturbations
            h_raw = np.random.uniform(0, 1, N)
            h_init = _project_simplex_box(h_raw, target_sum)
        
        # Optimization using L-BFGS-B
        # We need to handle the equality constraint. L-BFGS-B doesn't support
        # equality constraints directly. Instead, we can use SLSQP or reduce dimension.
        
        # Use SLSQP which supports constraints
        # But SLSQP can be slow. Let's try a different approach:
        # Use the projection step within the gradient.
        
        # Actually, let's use scipy.optimize.minimize with method='SLSQP'
        # or 'trust-constr' which can handle constraints.
        
        try:
            res = minimize(
                _objective_constrained,
                h_init,
                jac=_objective_constrained_grad,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_candidate = res.x
            # Project to ensure feasibility
            h_candidate = _project_simplex_box(h_candidate, target_sum)
            
            score = _overlap_max(h_candidate)
            if score < best_score:
                best_score = score
                best_h = h_candidate.copy()
                
        except Exception as e:
            pass
        
        # Time check
        if time.time() > deadline - 60:
            break
    
    # Local search refinement
    remaining = deadline - 10 - time.time()
    if remaining > 5:
        np.random.seed(123)
        step_size = 0.05
        last_improve = time.time()
        
        while time.time() < deadline - 10:
            noise = np.random.normal(0, step_size, N)
            h_new = best_h + noise
            h_new = _project_simplex_box(h_new, target_sum)
            
            score_new = _overlap_max(h_new)
            
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
                last_improve = time.time()
                step_size *= 0.95
            else:
                if time.time() - last_improve > 3:
                    step_size *= 1.05
                    step_size = min(step_size, 0.15)
    
    # Final projection and validation
    best_h = _project_simplex_box(best_h, target_sum)
    
    # Ensure sum is exactly target_sum within tolerance
    diff = target_sum - np.sum(best_h)
    if abs(diff) > 1e-6:
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
        best_h = _project_simplex_box(best_h, target_sum)
    
    return best_h


if __name__ == "__main__":
    start = time.time()
    h = construct()
    elapsed = time.time() - start
    
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    sum_h = np.sum(h)
    overlaps = (len(h) / 2.0) - autocorr
    max_overlap = np.max(overlaps)
    
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Sum: {sum_h:.10f}")
    print(f"Min h: {np.min(h):.6f}, Max h: {np.max(h):.6f}")
    print(f"Max overlap (raw): {max_overlap:.10f}")
    print(f"Score (upper bound on C5): {max_overlap:.10f}")
