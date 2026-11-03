"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
DOMAIN = np.linspace(-0.25, 0.25, N, endpoint=False) + 1.0/(2*N)

def _compute_ratio(heights):
    """Compute the normalized max autoconvolution ratio."""
    h = np.array(heights, dtype=np.float64)
    if np.all(h <= 0):
        return 1.0
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    l1_norm = np.sum(h)
    if l1_norm <= 0:
        return 1.0
    return max_conv / (l1_norm * l1_norm)

def _params_to_h(params, K):
    """
    Convert parameters to step heights.
    params layout: [c0, s0, a0, c1, s1, a1, ...]
    c: center (scaled to [-0.25, 0.25])
    s: std dev (log-scaled for stability, s = exp(p_s))
    a: amplitude (log-scaled, a = exp(p_a))
    """
    h = np.zeros(N)
    for i in range(K):
        c = params[3*i]
        s = np.exp(params[3*i + 1])
        a = np.exp(params[3*i + 2])
        # Gaussian: a * exp(-0.5 * ((x - c) / s)^2)
        h += a * np.exp(-0.5 * ((DOMAIN - c) / s) ** 2)
    # Ensure non-negative (Gaussian sum is positive if a>0)
    return h

def _objective(params, K):
    h = _params_to_h(params, K)
    return _compute_ratio(h)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # We will try a few different K values (number of Gaussians)
    # and a few random restarts for each.
    best_h = [1.0] * N
    best_ratio = _compute_ratio(best_h)
    
    # Start with a simple broad bump
    # params: [center, log_width, log_amp]
    # Center 0, width ~0.2, amp ~1
    init_params = [0.0, np.log(0.15), np.log(1.0)]
    
    # Try K=1, K=2, K=3
    for K in [1, 2, 3]:
        if time.time() > deadline - 10: # Leave time for final check
            break
            
        # Number of restarts depends on K
        n_restarts = 5 if K == 1 else (3 if K == 2 else 2)
        
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize parameters
            if r == 0:
                # Deterministic start: single central bump or symmetric pair
                if K == 1:
                    params = [0.0, np.log(0.15), np.log(1.0)]
                elif K == 2:
                    # Symmetric pair
                    params = [-0.1, np.log(0.08), np.log(0.7), 
                              0.1, np.log(0.08), np.log(0.7)]
                else:
                    # Three bumps: center, left, right
                    params = [0.0, np.log(0.05), np.log(0.5),
                              -0.15, np.log(0.05), np.log(0.5),
                              0.15, np.log(0.05), np.log(0.5)]
            else:
                # Random start
                # Centers in [-0.2, 0.2]
                # Widths in [0.01, 0.2] -> log
                # Amps in [0.1, 2.0] -> log
                params = []
                for i in range(K):
                    c = np.random.uniform(-0.2, 0.2)
                    s = np.random.uniform(np.log(0.01), np.log(0.2))
                    a = np.random.uniform(np.log(0.1), np.log(2.0))
                    params.extend([c, s, a])
            
            # Optimize
            # Nelder-Mead is robust for small dimensions without gradients
            # Max iterations limited to respect time budget
            max_iter = 200 if K <= 2 else 100
            
            try:
                res = minimize(_objective, params, args=(K,), method='Nelder-Mead',
                               options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if res.success or res.fun < best_ratio:
                    h = _params_to_h(res.x, K)
                    # Ensure non-negative (should be, but clip just in case of numerical issues)
                    h = np.clip(h, 0, None)
                    ratio = _compute_ratio(h)
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_h = h.tolist()
                        
            except Exception as e:
                pass
                
    # Final quick local refinement on the best found h using coordinate descent?
    # Or just return best_h.
    # Given time constraints, returning the best from Nelder-Mead is usually sufficient.
    # However, a simple grid search on the "scale" of the best h might help if normalization is off?
    # The ratio is scale-invariant: max(c*h) / (sum(h))^2 = max(c)*scale / (sum*h*scale)^2 = ratio / scale^2?
    # Wait.
    # Let h' = alpha * h.
    # conv(h') = alpha^2 * conv(h).
    # max(conv(h')) = alpha^2 * max(conv(h)).
    # sum(h') = alpha * sum(h).
    # ratio = (alpha^2 * max_conv) / (alpha * sum_h)^2 = max_conv / sum_h^2.
    # The ratio is scale invariant. So we don't need to optimize the overall scale.
    
    return best_h
# EVOLVE-BLOCK-END
