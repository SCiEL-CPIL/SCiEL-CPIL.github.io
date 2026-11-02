import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_ratio(h):
    """Compute ||h||^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return -1.0  # invalid
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    # Autoconvolution: h convolved with h
    conv = np.convolve(h, h, mode='full')
    conv_max = np.max(conv)
    if conv_max <= 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

def compute_ratio_smooth(h, alpha=100.0):
    """Compute ratio with smooth approximation of max via log-sum-exp."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return -1.0
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    if conv_max <= 0:
        return 0.0
    # Smooth max: (1/alpha) * log(sum(exp(alpha * conv)))
    conv_scaled = alpha * conv
    max_conv_scaled = np.max(conv_scaled)
    log_sum_exp = max_conv_scaled + np.log(np.sum(np.exp(conv_scaled - max_conv_scaled)))
    conv_max_smooth = log_sum_exp / alpha
    if conv_max_smooth <= 0:
        return 0.0
    return l2_sq / (l1 * conv_max_smooth)

def coordinate_polish(h, max_iter=500, tol=1e-10):
    """Coordinate descent polish: optimize each coordinate while fixing others."""
    h = np.array(h, dtype=float)
    best_score = compute_ratio(h)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try to optimize coordinate i
            h_old = h[i]
            # Grid search around current value
            for scale in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
                h[i] = h_old * scale
                s = compute_ratio(h)
                if s > best_score + tol:
                    best_score = s
                    improved = True
                else:
                    h[i] = h_old
            # Also try setting to 0
            h[i] = 0.0
            s = compute_ratio(h)
            if s > best_score + tol:
                best_score = s
                improved = True
            else:
                h[i] = h_old
        if not improved:
            break
    
    return h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Parameterize: use a smooth positive shape.
    # We optimize over log-space to ensure positivity.
    x = np.linspace(-1, 1, N)
    
    # Generate diverse starting points
    start_points = []
    
    # Triangular
    t = np.maximum(0, 1 - np.abs(x))
    start_points.append(np.log(np.maximum(t, 1e-10)))
    
    # Gaussian with various widths
    for sigma in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        g = np.exp(-x**2 / (2 * sigma**2))
        start_points.append(np.log(np.maximum(g, 1e-10)))
    
    # Uniform
    start_points.append(np.log(np.ones(N) * 1e-10))
    
    # Narrow triangular
    for width in [0.2, 0.3, 0.4, 0.5, 0.6]:
        t = np.maximum(0, 1 - np.abs(x) / width)
        start_points.append(np.log(np.maximum(t, 1e-10)))
    
    # Bump-like functions
    for p in [2, 4, 8]:
        bump = np.maximum(0, 1 - np.abs(x)**p)
        start_points.append(np.log(np.maximum(bump, 1e-10)))
    
    # Sparse-like: few non-zero entries
    for n_nonzero in [1, 2, 3, 5]:
        h_sparse = np.zeros(N)
        indices = np.random.choice(N, size=n_nonzero, replace=False)
        h_sparse[indices] = 1.0
        start_points.append(np.log(np.maximum(h_sparse, 1e-10)))
    
    # Optimize from each starting point
    best_score = 0.0
    best_h = None
    
    def neg_ratio(z):
        h = np.exp(z)
        r = compute_ratio_smooth(h, alpha=50.0)
        if r <= 0:
            return 0.0
        return -r
    
    for i, z0 in enumerate(start_points):
        if time.time() > deadline - 10.0:
            break
        
        try:
            # Use L-BFGS-B for better convergence
            res = minimize(neg_ratio, z0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            z_opt = res.x
            h_opt = np.exp(z_opt)
            score = compute_ratio(h_opt)
            
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            continue
    
    # Also try Nelder-Mead on best result
    if best_h is not None and time.time() < deadline - 5.0:
        try:
            z_best = np.log(np.maximum(best_h, 1e-15))
            res = minimize(neg_ratio, z_best, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            z_final = res.x
            h_final = np.exp(z_final)
            score_final = compute_ratio(h_final)
            if score_final > best_score:
                best_h = h_final
                best_score = score_final
        except Exception:
            pass
    
    # Coordinate polish
    if best_h is not None and time.time() < deadline - 3.0:
        try:
            h_polished, score_polished = coordinate_polish(best_h, max_iter=200)
            if score_polished > best_score:
                best_h = h_polished
                best_score = score_polished
        except Exception:
            pass
    
    # Scale: use a small but safe scale
    scale = 1e-6
    h_out = best_h * scale
    
    # Ensure non-negative
    h_out = np.maximum(h_out, 0.0)
    
    return h_out.tolist()
