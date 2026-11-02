import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
warnings.filterwarnings('ignore')

def _precompute_conv_matrix():
    """Precompute the autoconvolution kernel for step functions using vectorization."""
    dw = 1.0 / N
    x = np.linspace(-0.25, 0.25, N + 1)
    
    # Sample autoconvolution at fine grid points
    n_pts = 4 * N  # fine grid
    t_grid = np.linspace(-0.5, 0.5, n_pts + 1)
    midpoints = (t_grid[:-1] + t_grid[1:]) / 2.0
    
    # Vectorized computation of K
    xi = x[:-1].reshape(N, 1, 1)
    xi1 = x[1:].reshape(N, 1, 1)
    xj = x[:-1].reshape(1, N, 1)
    xj1 = x[1:].reshape(1, N, 1)
    t = midpoints.reshape(1, 1, -1)
    
    lo = np.maximum(xi, t - xj1)
    hi = np.minimum(xi1, t - xj)
    K = np.maximum(0, hi - lo)
    
    return K, midpoints

_K, _t_mid = _precompute_conv_matrix()
_K_flat = _K.reshape(N * N, -1)

def _compute_score(h):
    """Compute the ratio L2^2 / (L1 * sup|autoconv|) for step heights h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    
    # Scale invariance: we can normalize h to sum to 1 to improve conditioning
    h_norm = h / sum_h
    
    dw = 1.0 / N
    
    # L1 norm (integral) = 1 * dw (since sum h_norm = 1)
    L1 = 1.0 * dw
    L2_sq = np.sum(h_norm**2) * dw
    
    h_outer = np.outer(h_norm, h_norm).flatten()
    autoconv = h_outer @ _K_flat
    
    sup_conv = np.max(np.abs(autoconv))
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)

def _score_neg(h):
    """Negative score for minimization."""
    return -_compute_score(h)

def _spectral_waterfilling(h_init, max_iter=200, tol=1e-10):
    """
    Use spectral method to find the direction that maximizes the ratio.
    The autoconvolution is a quadratic form: h^T A h where A is the matrix 
    representing the supremum of the autoconvolution.
    
    Actually, the supremum is the max over t of |sum_{i,j} h_i h_j K_{ij}(t)|.
    This is not a simple quadratic form. But we can use an alternating approach:
    
    1. Fix the current autoconvolution profile, find the t that maximizes it.
    2. Then maximize h^T (K(t) vec) h / (h^T 1 * sup) ... this is getting complicated.
    
    Instead, let's use a different structural approach: 
    The problem is equivalent to maximizing ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    
    By scaling invariance, set ||h||_1 = 1. Then maximize ||h||_2^2 / ||h*h||_inf.
    
    This is a ratio of a convex function (||h||_2^2) to a convex function (||h*h||_inf).
    
    Let's try a different approach: use the fact that for a fixed support, 
    the optimal heights can be found by solving a generalized eigenvalue problem.
    
    Actually, let's try a coordinate descent / alternating approach:
    1. For each coordinate i, given the others fixed, find the optimal h_i.
    2. The autoconvolution is a quadratic form in h.
    """
    h = np.array(h_init, dtype=float)
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return h
    
    for it in range(max_iter):
        h_old = h.copy()
        for i in range(N):
            # Fix all other coordinates, optimize h_i
            # The score is (sum_j h_j^2) / (sum_j h_j * sup_t |sum_{a,b} h_a h_b K_ab(t)|)
            # This is complex. Let's just do a 1D search for h_i.
            
            # Sample h_i values
            h_i_vals = np.linspace(0, 10, 50)
            best_val = -np.inf
            best_hi = h[i]
            
            for hi in h_i_vals:
                h_new = h.copy()
                h_new[i] = hi
                sc = _compute_score(h_new)
                if sc > best_val:
                    best_val = sc
                    best_hi = hi
            
            h[i] = best_hi
        
        # Check convergence
        if np.max(np.abs(h - h_old)) < tol:
            break
    
    return h

def _random_sparse_start(n_peaks):
    """Generate a random sparse start with n_peaks non-zero entries."""
    h = np.zeros(N)
    idx = np.random.choice(N, n_peaks, replace=False)
    h[idx] = np.random.uniform(0.5, 2.0, n_peaks)
    return h

def _alternating_pattern(offset=0):
    """Generate an alternating pattern start."""
    h = np.zeros(N)
    h[offset::2] = 1.0
    return h

def _gaussian_bump(center, width=5):
    """Generate a Gaussian bump start."""
    x = np.arange(N)
    h = np.exp(-0.5 * ((x - center) / width) ** 2)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = -np.inf
    
    # Generate diverse random starts with different structural families
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(N))
    
    # 2. Sparse with varying sparsity
    for n_peaks in [3, 5, 7, 10]:
        for _ in range(3):
            starts.append(_random_sparse_start(n_peaks))
    
    # 3. Alternating patterns
    for offset in [0, 1]:
        starts.append(_alternating_pattern(offset))
    
    # 4. Gaussian bumps at various positions
    for center in [5, 15, 25, 35, 45]:
        for width in [3, 5, 8]:
            starts.append(_gaussian_bump(center, width))
    
    # 5. Two-bump patterns
    for c1 in range(5, N-5, 10):
        for c2 in range(c1+10, N, 10):
            h = np.zeros(N)
            h += _gaussian_bump(c1, 4)
            h += _gaussian_bump(c2, 4)
            starts.append(h)
    
    # 6. Random exponential
    for _ in range(10):
        starts.append(np.random.exponential(1.0, N))
    
    # 7. Cosine patterns
    for freq in [1, 2, 3]:
        x = np.linspace(0, 2*np.pi*freq, N)
        starts.append(np.maximum(np.cos(x), 0))
    
    for h0 in starts:
        if time.time() > deadline - 30:
            break
        
        h0 = np.maximum(h0, 0.0)
        
        # Try spectral waterfilling (coordinate descent)
        try:
            h_trial = _spectral_waterfilling(h0, max_iter=50, tol=1e-8)
            
            # Polish with L-BFGS-B
            if time.time() < deadline - 15:
                result = minimize(_score_neg, h_trial, method='L-BFGS-B',
                                 bounds=[(0, 10)] * N,
                                 options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                h_trial = np.maximum(result.x, 0.0)
            
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
                
        except Exception:
            pass
    
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return list(best_h)
