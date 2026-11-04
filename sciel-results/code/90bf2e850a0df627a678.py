import numpy as np
from scipy.optimize import minimize

N = 50
w = 1.0 / N

# Precompute the matrix M for fast autoconvolution peak calculation
# m ranges from 0 to 2N
# t ranges from 0 to 2N-2
# M[m, t] = max(0, 1 - |m - (t+1)|)
m_idx = np.arange(2 * N + 1)
t_idx = np.arange(2 * N - 1)
# t+1 ranges from 1 to 2N-1
# diff = m - (t+1)
M = np.maximum(0, 1 - np.abs(m_idx[:, None] - (t_idx + 1)[None, :]))

def _autoconv_peak(h):
    """
    Compute the supremum norm of the autoconvolution using the precomputed matrix M.
    h: array of non-negative heights.
    """
    # Discrete auto-convolution C
    # C[t] = sum_i h_i h_{t-i}
    # Using np.convolve
    C = np.convolve(h, h, mode='full')
    # C has length 2N-1
    # g = M @ C
    g = M @ C
    return np.max(g)

def construct():
    def objective(h_flat):
        h = np.maximum(0, h_flat)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 1e6
        l2_sq = np.sum(h**2)
        peak = _autoconv_peak(h)
        if peak < 1e-12:
            return 1e6
        ratio = l2_sq / (l1 * peak)
        return -ratio

    # Initial guess: uniform distribution
    # This should give a ratio of ~2.0
    h0 = np.ones(N)
    
    # Bounds: [0, inf)
    bounds = [(0, None)] * N
    
    # Use L-BFGS-B or TRF
    # TRF (Trust Region Reflective) is good for bounded problems
    try:
        res = minimize(objective, h0, method='trf', bounds=bounds, 
                       options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-10})
        h_opt = np.maximum(0, res.x)
        
        # Also try L-BFGS-B from the same start just in case
        res2 = minimize(objective, h0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        h_opt2 = np.maximum(0, res2.x)
        
        # Pick the better one
        if objective(h_opt) < objective(h_opt2):
            h_final = h_opt
        else:
            h_final = h_opt2
            
    except Exception:
        h_final = h0
        
    # Normalize to ensure non-negative and reasonable scale
    s = np.sum(h_final)
    if s > 0:
        h_final = h_final / s
    else:
        h_final = h0
        
    return h_final
