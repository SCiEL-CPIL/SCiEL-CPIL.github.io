import numpy as np
from scipy import optimize
import time

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    if len(h) == 0:
        return 0.0
    total = np.sum(h)
    if total < 1e-12:
        return 0.0
    conv = np.convolve(h, h, 'full')
    return float(np.max(conv) / (total * total))

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin for the 150s limit
    
    # We want to minimize max(convolve(h,h,'full')) / (sum(h))^2
    # This is scale-invariant. We can fix sum(h) = 1 without loss of generality
    # and minimize max(convolve(h,h)).
    # 
    # Strategy: Use a smooth surrogate (soft-max via log-sum-exp) and optimize
    # with L-BFGS-B, then polish with a more aggressive approach.
    
    def soft_max_surrogate(h, T):
        """Soft approximation to max(conv) for smoothness."""
        conv = np.convolve(h, h, 'full')
        cmax = np.max(conv)
        # Numerically stable log-sum-exp
        shifted = (conv - cmax) / T
        # Clip to avoid overflow in exp
        shifted = np.clip(shifted, -50, 50)
        return cmax + T * np.log(np.sum(np.exp(shifted)))
    
    def objective(x, T):
        """Objective: soft-max of autoconvolution, with h = |x| and normalized to sum=1."""
        h = np.abs(x)
        s = np.sum(h)
        if s < 1e-12:
            return 1e10
        h = h / s
        return soft_max_surrogate(h, T)
    
    def exact_objective(x):
        """Exact objective: max of autoconvolution, normalized."""
        h = np.abs(x)
        s = np.sum(h)
        if s < 1e-12:
            return 1e10
        h = h / s
        conv = np.convolve(h, h, 'full')
        return float(np.max(conv))
    
    best_h = np.ones(N) / N
    best_score = compute_autoconv_max(best_h)
    
    # Multiple restarts with different initializations and temperatures
    np.random.seed(42)
    
    initializations = [
        np.ones(N) / N,  # uniform
        1.0 - np.abs(np.linspace(-1, 1, N)),  # triangle
        np.abs(np.linspace(-1, 1, N)) + 0.05,  # inverse triangle
        0.5 * (1 + np.cos(np.pi * np.linspace(-1, 1, N))),  # cosine bump
        np.random.rand(N) + 0.05,  # random
    ]
    
    temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for restart_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        h0 = np.abs(h0)
        s = np.sum(h0)
        if s < 1e-12:
            continue
        h0 = h0 / s
        
        # Try a few temperatures, starting coarse and refining
        for T in temperatures:
            if time.time() > deadline - 15:
                break
            
            try:
                result = optimize.minimize(
                    lambda x: objective(x, T),
                    h0,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                h_new = np.abs(result.x)
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new / s_new
                
                score_new = compute_autoconv_max(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
                    h0 = h_new.copy()  # Use improved solution for next temperature
                    
            except Exception:
                pass
    
    # Final polish: use the best solution and do a very tight optimization
    if time.time() < deadline - 10:
        try:
            h_polish = best_h.copy()
            result = optimize.minimize(
                lambda x: objective(x, 0.0001),
                h_polish,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_new = np.abs(result.x)
            s_new = np.sum(h_new)
            if s_new > 1e-12:
                h_new = h_new / s_new
                score_new = compute_autoconv_max(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
        except Exception:
            pass
    
    # Additional local refinement: try small perturbations around the best solution
    if time.time() < deadline - 5:
        try:
            for _ in range(3):
                h_pert = best_h + np.random.randn(N) * 0.001
                h_pert = np.maximum(h_pert, 0)
                s_pert = np.sum(h_pert)
                if s_pert < 1e-12:
                    continue
                h_pert = h_pert / s_pert
                
                result = optimize.minimize(
                    lambda x: objective(x, 0.0005),
                    h_pert,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                h_new = np.abs(result.x)
                s_new = np.sum(h_new)
                if s_new > 1e-12:
                    h_new = h_new / s_new
                    score_new = compute_autoconv_max(h_new)
                    if score_new < best_score:
                        best_score = score_new
                        best_h = h_new.copy()
        except Exception:
            pass
    
    return list(best_h)
