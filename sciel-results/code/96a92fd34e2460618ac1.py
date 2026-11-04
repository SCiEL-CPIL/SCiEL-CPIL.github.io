import numpy as np
from scipy.optimize import minimize

N = 50

# Precompute centers and sum_centers for the autoconvolution
w = 1.0 / N
centers = -0.25 + (np.arange(N) + 0.5) * w
sum_centers = centers[:, None] + centers[None, :]
sum_centers_flat = sum_centers.flatten()

def _autoconv_peak(h):
    """
    Compute the supremum norm of the autoconvolution of a piecewise-constant
    function defined by heights h on N equal intervals over [-1/4, 1/4].
    """
    hh = np.outer(h, h)
    
    max_val = 0.0
    for x in sum_centers_flat:
        diff = np.abs(x - sum_centers)
        tri = np.maximum(0, 1 - diff / w)
        val = np.sum(hh * tri)
        if val > max_val:
            max_val = val
            
    return max_val

def construct():
    # Objective function to minimize: - (||h||_2^2 / (||h||_1 * ||h*h||_inf))
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

    # Use multiple initial guesses to escape local minima
    best_h = None
    best_val = -1e6
    
    # Initial guess 1: single spike in the middle
    h0 = np.zeros(N)
    h0[N // 2] = 1.0
    
    # Initial guess 2: two spikes
    h0_2 = np.zeros(N)
    h0_2[N // 4] = 0.5
    h0_2[3 * N // 4] = 0.5
    
    # Initial guess 3: uniform
    h0_3 = np.ones(N) / N
    
    # Initial guess 4: alternating
    h0_4 = np.zeros(N)
    h0_4[::2] = 1.0 / (N // 2)
    
    for h0 in [h0, h0_2, h0_3, h0_4]:
        bounds = [(0, None)] * N
        try:
            res = minimize(objective, h0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            h_opt = np.maximum(0, res.x)
            val = -objective(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt
        except Exception:
            pass
    
    if best_h is None:
        best_h = h0
    
    # Normalize to ensure non-negative and reasonable scale
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = h0
        
    return best_h
