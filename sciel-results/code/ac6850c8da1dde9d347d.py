"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 50

def _compute_ratio(h):
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)  # enforce nonnegativity
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    if l1 == 0:
        return 0.0
    # Compute autoconvolution
    n = len(h)
    conv = np.correlate(h, h, mode='full')
    conv_max = np.max(conv)
    if conv_max == 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

def _neg_ratio(h):
    return -_compute_ratio(h)

# EVOLVE-BLOCK-START
def construct():
    # Try multiple initializations and pick the best
    best_h = None
    best_score = 0.0
    
    # Different initializations
    inits = []
    
    # Gaussian centered
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-2*x**2))
    inits.append(np.exp(-5*x**2))
    inits.append(np.exp(-1*x**2))
    
    # Triangular
    inits.append(np.maximum(0, 1 - np.abs(x)))
    
    # Uniform (baseline)
    inits.append(np.ones(N))
    
    # Two peaks
    inits.append(np.exp(-20*(x-0.3)**2) + np.exp(-20*(x+0.3)**2))
    inits.append(np.exp(-15*(x-0.4)**2) + np.exp(-15*(x+0.4)**2))
    
    # Sparse-like
    inits.append(np.where(np.abs(x) < 0.3, 1.0, 0.0))
    
    for init in inits:
        init = np.maximum(init, 0)
        # Normalize to avoid numerical issues
        init = init / np.max(init) if np.max(init) > 0 else init
        
        # Use L-BFGS-B for bounded optimization
        bounds = [(0, None)] * N
        result = minimize(_neg_ratio, init, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        h_opt = np.maximum(result.x, 0)
        score = _compute_ratio(h_opt)
        
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Convert to list of floats
    if best_h is None:
        return [1.0] * N
    
    return [float(v) for v in best_h]
# EVOLVE-BLOCK-END
