import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).
    
    Uses a smooth epigraph formulation: introduce auxiliary variable t >= 0
    and minimize t subject to t >= C(k) for all shifts k, where
    C(k) = sum(h) - autocorr(k) is the overlap at shift k.
    
    We use log-sum-exp to create a smooth surrogate for the max, then
    polish with a final local search.
    """
    deadline = time.time() + 145  # Safety margin
    
    N = 100
    
    def autocorr(h):
        """Compute circular autocorrelation via FFT."""
        N = len(h)
        F = np.fft.fft(h)
        ac = np.real(np.fft.ifft(F * np.conj(F)))[:N]
        return ac
    
    def compute_C(h):
        """Compute overlap values C(k) for all non-zero shifts."""
        N = len(h)
        s = h.sum()
        ac = autocorr(h)
        C_vals = s - ac[1:]
        return C_vals
    
    def max_overlap(h):
        """Compute the maximum overlap over non-zero shifts."""
        C_vals = compute_C(h)
        return np.max(C_vals)
    
    def smooth_max(h, beta=50.0):
        """
        Smooth surrogate for max(C(k)) using log-sum-exp.
        As beta -> inf, this approaches the true max.
        """
        C_vals = compute_C(h)
        # Log-sum-exp for numerical stability
        C_max = np.max(C_vals)
        if C_max < -1000:  # Underflow guard
            return 0.0
        exp_C = np.exp(beta * (C_vals - C_max))
        return C_max + np.log(np.sum(exp_C)) / beta
    
    def objective(params):
        """
        Objective: minimize smooth max overlap.
        params: h vector of length N.
        """
        h = params
        return smooth_max(h)
    
    def objective_with_penalty(params, penalty_coeff=100.0):
        """
        Objective with penalty for constraint violations:
        - sum(h) should be N/2
        - 0 <= h[i] <= 1
        """
        h = params
        base = smooth_max(h)
        
        # Penalty for sum constraint
        sum_diff = h.sum() - N / 2.0
        penalty_sum = penalty_coeff * sum_diff ** 2
        
        # Penalty for bound violations
        below_zero = np.maximum(0, -h)
        above_one = np.maximum(0, h - 1.0)
        penalty_bounds = penalty_coeff * (np.sum(below_zero ** 2) + np.sum(above_one ** 2))
        
        return base + penalty_sum + penalty_bounds
    
    # Initialize with constant 0.5
    h0 = np.full(N, 0.5)
    
    # Use L-BFGS-B with bounds
    bounds = [(0.0, 1.0)] * N
    constraints = [
        {'type': 'eq', 'fun': lambda x: x.sum() - N / 2.0, 
         'jac': lambda x: np.ones(N)}
    ]
    
    best_h = h0.copy()
    best_val = max_overlap(h0)
    
    # Phase 1: Coarse optimization with lower beta
    beta_coarse = 20.0
    smooth_max.__defaults__ = (beta_coarse,)
    
    try:
        result = minimize(
            objective,
            h0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.fun < best_val or result.success:
            # Project back to feasible set
            h_opt = result.x
            h_opt = np.clip(h_opt, 0, 1)
            diff = N / 2 - h_opt.sum()
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            # Re-fix sum
            for _ in range(5):
                diff = N / 2 - h_opt.sum()
                if abs(diff) < 1e-12:
                    break
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0, 1)
            
            val = max_overlap(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_h
    
    # Phase 2: Fine optimization with higher beta
    beta_fine = 100.0
    smooth_max.__defaults__ = (beta_fine,)
    
    try:
        result = minimize(
            objective,
            best_h,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.fun < best_val or result.success:
            h_opt = result.x
            h_opt = np.clip(h_opt, 0, 1)
            diff = N / 2 - h_opt.sum()
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            for _ in range(5):
                diff = N / 2 - h_opt.sum()
                if abs(diff) < 1e-12:
                    break
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0, 1)
            
            val = max_overlap(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_h
    
    # Phase 3: Local search polish
    rng = np.random.default_rng(42)
    h = best_h.copy()
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Random pairwise perturbation
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
        
        delta = rng.uniform(-0.02, 0.02)
        h_trial = h.copy()
        
        if 0 <= h_trial[i] + delta <= 1 and 0 <= h_trial[j] - delta <= 1:
            h_trial[i] += delta
            h_trial[j] -= delta
            val = max_overlap(h_trial)
            if val < best_val:
                best_val = val
                best_h = h_trial.copy()
                h = h_trial.copy()
    
    # Final validation
    N_final = len(best_h)
    best_h = np.clip(best_h, 0, 1)
    diff = N_final / 2 - best_h.sum()
    best_h += diff / N_final
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
