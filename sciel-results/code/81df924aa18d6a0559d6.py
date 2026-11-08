import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_norm = np.max(conv)
    if sup_norm == 0:
        return 0.0
    return l2_sq / (l1 * sup_norm)

# EVOLVE-BLOCK-START
def construct():
    import time
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # The ratio R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # We want to maximize this. Key insight: use optimization to find non-trivial shapes.
    # 
    # For nonnegative h, we can parameterize by shape (normalized) and scale.
    # Since R is homogeneous of degree -1, if the evaluator uses raw values,
    # we need to choose a reasonable scale. Let's assume max(h)=1 is expected.
    #
    # Structural change: Use scipy.optimize to find optimal shape with
    # a concentrated but multi-point profile.
    
    from scipy.optimize import minimize
    
    def neg_ratio(h):
        """Negative ratio for minimization."""
        h = np.maximum(h, 0)  # enforce nonnegativity
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        conv = np.convolve(h, h, mode='full')
        sup_norm = np.max(conv)
        if sup_norm < 1e-10:
            return 0.0
        return -(l2_sq / (l1 * sup_norm))
    
    # Try multiple initializations with different structures
    best_h = None
    best_score = -np.inf
    
    # Initialization 1: Single spike (baseline)
    h0 = np.zeros(N)
    h0[N//2] = 1.0
    res = minimize(neg_ratio, h0, method='L-BFGS-B', 
                   bounds=[(0, 1)]*N, options={'maxiter': 500, 'ftol': 1e-12})
    h_opt = np.maximum(res.x, 0)
    score = -res.fun
    if score > best_score:
        best_score = score
        best_h = h_opt.copy()
    
    # Check time
    if time.time() - t0 > deadline:
        h_out = best_h.copy()
        h_out[h_out < 1e-10] = 0.0
        return list(h_out)
    
    # Initialization 2: Gaussian profile
    x = np.linspace(-1, 1, N)
    h0 = np.exp(-2 * x**2)
    h0 /= np.max(h0)
    res = minimize(neg_ratio, h0, method='L-BFGS-B',
                   bounds=[(0, 1)]*N, options={'maxiter': 500, 'ftol': 1e-12})
    h_opt = np.maximum(res.x, 0)
    score = -res.fun
    if score > best_score:
        best_score = score
        best_h = h_opt.copy()
    
    # Check time
    if time.time() - t0 > deadline:
        h_out = best_h.copy()
        h_out[h_out < 1e-10] = 0.0
        return list(h_out)
    
    # Initialization 3: Triangular profile
    h0 = np.maximum(0, 1 - 2*np.abs(x))
    h0 /= np.max(h0)
    res = minimize(neg_ratio, h0, method='L-BFGS-B',
                   bounds=[(0, 1)]*N, options={'maxiter': 500, 'ftol': 1e-12})
    h_opt = np.maximum(res.x, 0)
    score = -res.fun
    if score > best_score:
        best_score = score
        best_h = h_opt.copy()
    
    # Check time
    if time.time() - t0 > deadline:
        h_out = best_h.copy()
        h_out[h_out < 1e-10] = 0.0
        return list(h_out)
    
    # Initialization 4: Two-peak profile
    h0 = np.zeros(N)
    h0[N//4] = 1.0
    h0[3*N//4] = 1.0
    res = minimize(neg_ratio, h0, method='L-BFGS-B',
                   bounds=[(0, 1)]*N, options={'maxiter': 500, 'ftol': 1e-12})
    h_opt = np.maximum(res.x, 0)
    score = -res.fun
    if score > best_score:
        best_score = score
        best_h = h_opt.copy()
    
    # Check time
    if time.time() - t0 > deadline:
        h_out = best_h.copy()
        h_out[h_out < 1e-10] = 0.0
        return list(h_out)
    
    # Initialization 5: Random perturbation of spike
    np.random.seed(42)
    h0 = np.zeros(N)
    h0[N//2] = 1.0
    h0 += np.random.randn(N) * 0.1
    h0 = np.maximum(h0, 0)
    h0 /= np.max(h0)
    res = minimize(neg_ratio, h0, method='L-BFGS-B',
                   bounds=[(0, 1)]*N, options={'maxiter': 500, 'ftol': 1e-12})
    h_opt = np.maximum(res.x, 0)
    score = -res.fun
    if score > best_score:
        best_score = score
        best_h = h_opt.copy()
    
    # Return best result, normalized to max=1
    h_out = best_h.copy()
    h_out[h_out < 1e-10] = 0.0
    if np.max(h_out) > 0:
        h_out /= np.max(h_out)
    else:
        h_out[N//2] = 1.0
    
    return list(h_out)
# EVOLVE-BLOCK-END
