"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution

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
    K = 4  # Number of Gaussian bumps
    X_GRID = np.linspace(-0.25, 0.25, N)
    
    # Parameters: [a1, c1, w1, a2, c2, w2, ...]
    # Bounds for each parameter type
    a_bounds = (0.0, 10.0)
    c_bounds = (-0.25, 0.25)
    w_bounds = (0.01, 0.5)
    
    bounds = []
    for _ in range(K):
        bounds.append(a_bounds)
        bounds.append(c_bounds)
        bounds.append(w_bounds)
        
    def params_to_heights(params):
        h = np.zeros(N)
        for i in range(K):
            a = params[3*i]
            c = params[3*i+1]
            w = params[3*i+2]
            # Gaussian: a * exp(-(x-c)^2 / (2*w^2))
            h += a * np.exp(-(X_GRID - c)**2 / (2 * w**2))
        return h
        
    def neg_ratio_params(params):
        h = params_to_heights(params)
        return -_compute_ratio(h)

    # Use Differential Evolution for global search over the low-dimensional parameter space
    # This is robust to local minima in the parameter space
    de_result = differential_evolution(
        neg_ratio_params, 
        bounds=bounds, 
        seed=42, 
        maxiter=100, 
        tol=1e-6,
        popsize=15,
        polish=True
    )
    
    best_params = de_result.x
    best_h = params_to_heights(best_params)
    best_score = _compute_ratio(best_h)
    
    # Local refinement on the full 50-dimensional space
    # Start from the best parametric solution
    # This allows the solution to deviate from the strict Gaussian sum structure
    bounds_50 = [(0, None)] * N
    local_result = minimize(
        _neg_ratio, 
        best_h, 
        method='L-BFGS-B', 
        bounds=bounds_50,
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    local_h = np.maximum(local_result.x, 0)
    local_score = _compute_ratio(local_h)
    
    if local_score > best_score:
        best_h = local_h
        best_score = local_score
        
    # Try a few more local starts from slightly perturbed parametric solutions
    # to ensure we don't get stuck
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Perturb the best params
        perturbed_params = best_params + rng.normal(0, 0.1, size=len(best_params))
        # Clip to bounds
        for i in range(K):
            perturbed_params[3*i] = np.clip(perturbed_params[3*i], 0, 10)
            perturbed_params[3*i+1] = np.clip(perturbed_params[3*i+1], -0.25, 0.25)
            perturbed_params[3*i+2] = np.clip(perturbed_params[3*i+2], 0.01, 0.5)
            
        pert_h = params_to_heights(perturbed_params)
        pert_local = minimize(
            _neg_ratio, 
            pert_h, 
            method='L-BFGS-B', 
            bounds=bounds_50,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        pert_h_opt = np.maximum(pert_local.x, 0)
        pert_score = _compute_ratio(pert_h_opt)
        
        if pert_score > best_score:
            best_score = pert_score
            best_h = pert_h_opt

    if best_h is None:
        return [1.0] * N
        
    return [float(v) for v in best_h]
# EVOLVE-BLOCK-END
