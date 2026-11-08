import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 50

def _compute_ratio(h):
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    if l1 == 0:
        return 0.0
    conv = np.correlate(h, h, mode='full')
    conv_max = np.max(conv)
    if conv_max == 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

def _neg_ratio_params(params):
    # params: [A1, c1, w1, A2, c2, w2, A3, c3, w3]
    # Generate heights
    x = np.linspace(-1, 1, N)
    h = np.zeros(N)
    for i in range(3):
        A = params[i*3]
        c = params[i*3+1]
        w = params[i*3+2]
        # Top-hat or Gaussian? "Top-hat" is specified.
        # Let's use a smooth top-hat (raised cosine) to avoid discontinuities in gradients if we used grad, but DE doesn't need grads.
        # Actually, strict top-hat: 1 if |x-c| < w/2 else 0.
        # This is non-smooth. DE handles it fine.
        # But maybe a Gaussian is better for optimization? The prompt says "top-hat".
        # Let's use a smooth approximation of top-hat: sigmoid or just a box.
        # Box:
        h += A * (np.abs(x - c) < w/2)
    
    # Ensure non-negative (A is bounded >= 0)
    return -_compute_ratio(h)

def construct():
    # Optimize 3 top-hats
    # Bounds: A in [0, 5], c in [-1, 1], w in [0.05, 2.0]
    bounds = [(0, 5), (-1, 1), (0.05, 2.0)] * 3
    
    # Use differential evolution for global search
    # seed for reproducibility? No, random is fine, but we want consistent results.
    # The evaluator runs it once.
    result = differential_evolution(
        _neg_ratio_params, 
        bounds, 
        seed=42, 
        maxiter=100, 
        popsize=15, 
        tol=1e-6,
        polish=True
    )
    
    best_params = result.x
    # Reconstruct h
    x = np.linspace(-1, 1, N)
    h = np.zeros(N)
    for i in range(3):
        A = best_params[i*3]
        c = best_params[i*3+1]
        w = best_params[i*3+2]
        h += A * (np.abs(x - c) < w/2)
        
    # Refine with local optimization on the discrete heights?
    # The parametric solution might be stuck in a local min of the parametric space.
    # Let's take the best h from DE and refine it with L-BFGS-B on the 50 dimensions.
    
    def _neg_ratio_h(h):
        return -_compute_ratio(h)
        
    bounds_h = [(0, None)] * N
    result2 = minimize(_neg_ratio_h, h, method='L-BFGS-B', bounds=bounds_h,
                       options={'maxiter': 500, 'ftol': 1e-10})
    
    h_final = np.maximum(result2.x, 0)
    score = _compute_ratio(h_final)
    
    return [float(v) for v in h_final]
