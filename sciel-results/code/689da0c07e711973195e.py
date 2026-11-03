import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the score for a given height vector h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # We want to maximize L2^2 / (L1 * Sup(AutoConv))
    # This is equivalent to maximizing L2^2 / Sup(AutoConv) subject to L1 = 1
    # Let's use a different approach: treat this as a non-convex optimization problem
    # and use multiple strategies including:
    # 1. Random restarts with local optimization
    # 2. Pattern-based initializations (periodic, clustered, etc.)
    # 3. Gradient-based optimization with multiple starting points
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Strategy 1: Various parametric families
    parametric_inits = []
    
    # Uniform
    parametric_inits.append(np.ones(N) / N)
    
    # Gaussian
    for center in np.linspace(10, N-10, 5):
        for width in np.linspace(2, N/3, 5):
            h_gauss = np.exp(-0.5 * ((x_idx - center) / width)**2)
            h_gauss /= np.sum(h_gauss)
            parametric_inits.append(h_gauss)
            
    # Triangular
    for center in np.linspace(10, N-10, 5):
        for width in np.linspace(3, N/2, 5):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - center) / width)
            if np.sum(h_tri) > 0:
                h_tri /= np.sum(h_tri)
                parametric_inits.append(h_tri)
                
    # Periodic patterns (to minimize autoconvolution overlap)
    for period in [2, 3, 4, 5, 7, 10]:
        h_per = np.zeros(N)
        h_per[::period] = 1.0
        if np.sum(h_per) > 0:
            h_per /= np.sum(h_per)
            parametric_inits.append(h_per)
            
    # Alternating patterns
    for shift in range(min(5, N)):
        h_alt = np.zeros(N)
        h_alt[shift::2] = 1.0
        if np.sum(h_alt) > 0:
            h_alt /= np.sum(h_alt)
            parametric_inits.append(h_alt)
            
    # Clustered patterns
    for n_clusters in [2, 3, 4, 5]:
        cluster_size = max(1, N // (n_clusters * 2))
        h_cluster = np.zeros(N)
        for i in range(n_clusters):
            start = int(i * N / n_clusters)
            end = min(start + cluster_size, N)
            h_cluster[start:end] = 1.0
        if np.sum(h_cluster) > 0:
            h_cluster /= np.sum(h_cluster)
            parametric_inits.append(h_cluster)
            
    # Evaluate parametric inits
    for h_init in parametric_inits:
        s = _score_raw(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()
    
    # Strategy 2: Local optimization from multiple starting points
    # Objective: minimize negative score with L1 constraint
    
    def neg_score_with_constraint(h):
        """Negative of the objective, with implicit L1 normalization."""
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        h_norm = h / l1
        l2_sq = np.sum(h_norm**2)
        h_conv = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return 0.0
        return -(l2_sq / max_conv)
    
    # Constraints and bounds for SLSQP
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
    ]
    bounds = [(0.0, 1.0)] * N
    
    # Optimization function
    def optimize_from(h_init):
        """Optimize from a given starting point."""
        h_init = np.asarray(h_init, dtype=np.float64)
        l1 = np.sum(h_init)
        if l1 == 0:
            return None, 0.0
        h_start = h_init / l1
        
        try:
            res = minimize(neg_score_with_constraint, h_start, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                h_opt = res.x
                s = _score_raw(h_opt)
                return h_opt, s
        except Exception:
            pass
        return None, 0.0
    
    # Optimize from parametric inits
    for h_init in parametric_inits[:20]:  # Limit to avoid timeout
        if time.time() > deadline - 20:
            break
        h_opt, s = optimize_from(h_init)
        if h_opt is not None and s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Strategy 3: Random restarts
    n_random = 30
    for i in range(n_random):
        if time.time() > deadline - 15:
            break
        
        # Generate random initialization
        h_rand = np.random.rand(N)
        # Sometimes make it sparse
        if np.random.rand() < 0.3:
            mask = np.random.rand(N) < 0.5
            h_rand[mask] = np.random.rand(np.sum(mask))
            h_rand[~mask] = 0
        
        h_opt, s = optimize_from(h_rand)
        if h_opt is not None and s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Strategy 4: Final polish
    if time.time() < deadline - 5:
        h_opt, s = optimize_from(best_h)
        if h_opt is not None and s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    return list(best_h)

# EVOLVE-BLOCK-END
