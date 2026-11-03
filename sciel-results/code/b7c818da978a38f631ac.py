"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
X = np.linspace(-0.25, 0.25, N)
DX = X[1] - X[0]

# Precompute autoconvolution kernel indices
# autoconv[i,j] = 1 if X[i] - X[j] is in [-0.5, 0.5] (always true for our range)
# Actually, the autoconvolution of step function h(x) on [-1/4, 1/4] with itself:
# (h*h)(z) = integral h(t)h(z-t) dt
# For step heights on equal intervals, this becomes a discrete sum.
# Let's think more carefully.

# The task says: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
# Let h be the step function with heights h_0, ..., h_{49} on intervals of length DX.
# L2 norm squared = sum_i h_i^2 * DX
# L1 norm = sum_i h_i * DX
# Autoconvolution: (h*h)(z) = sum_i h_i * h_j * DX^2 where X_i - X_j ≈ z
# More precisely, for step functions, (h*h)(z) = sum_i h_i * h_{i-k} * DX^2 where z = k*DX
# The supremum norm of autoconvolution = max_k |sum_i h_i * h_{i-k}| * DX^2

# So we want to maximize: (sum h_i^2 * DX) / ((sum h_i * DX) * max_k |sum_i h_i * h_{i-k}| * DX^2)
# = (sum h_i^2) / ((sum h_i) * max_k |sum_i h_i * h_{i-k}| * DX)

# Since DX is constant, we maximize: (sum h_i^2) / ((sum h_i) * max_k |sum_i h_i * h_{i-k}|)

# Let's define:
# f(h) = (sum h_i^2) / ((sum h_i) * max_k |sum_i h_i * h_{i-k}|)

# We need h_i >= 0.

# Precompute the autoconvolution matrix
# For each lag k, autoconv[k] = sum_i h_i * h_{i-k} (with appropriate bounds)
# Since X is in [-1/4, 1/4], the autoconvolution is defined on [-1/2, 1/2]
# The lags k range from -(N-1) to (N-1)

def compute_autoconv(h):
    """Compute autoconvolution values for all lags."""
    h = np.asarray(h)
    n = len(h)
    autoconv = np.zeros(2*n-1)
    for k in range(-(n-1), n):
        if k >= 0:
            # sum_{i=k}^{n-1} h[i] * h[i-k]
            if k < n:
                autoconv[k + n - 1] = np.dot(h[k:], h[:n-k])
        else:
            # k < 0, let j = -k
            j = -k
            autoconv[k + n - 1] = np.dot(h[:n-j], h[j:])
    return autoconv

def objective_and_grad(h):
    """Compute the objective and its gradient.
    
    f(h) = (sum h_i^2) / ((sum h_i) * max_k |autoconv_k|)
    """
    h = np.asarray(h)
    n = len(h)
    sum_h2 = np.dot(h, h)
    sum_h = np.sum(h)
    autoconv = compute_autoconv(h)
    max_ac = np.max(np.abs(autoconv))
    
    if max_ac < 1e-15:
        return 0.0, np.zeros_like(h)
    
    f = sum_h2 / (sum_h * max_ac)
    
    # Gradient computation
    # f = sum_h2 / (sum_h * max_ac)
    # df/dh_i = (2*h_i * sum_h * max_ac - sum_h2 * (delta_i * max_ac + sum_h * dmax_ac/dh_i)) / (sum_h * max_ac)^2
    # where delta_i = 1 if h_i is in the argmax of |autoconv_k|
    
    # This is complex. Let's use numerical gradient or a simpler approach.
    # For now, let's just return the objective and let SLSQP compute the gradient numerically.
    return f

def score(h):
    """Compute the score (the value we want to maximize)."""
    h = np.asarray(h)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    n = len(h)
    sum_h2 = np.dot(h, h)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    autoconv = compute_autoconv(h)
    max_ac = np.max(np.abs(autoconv))
    if max_ac < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * max_ac)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Generate initial candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Alternating patterns
    for period in [2, 3, 4, 5]:
        for phase in range(period):
            h = np.array([1.0 if (i + phase) % period == 0 else 0.0 for i in range(N)])
            if np.sum(h) > 0:
                candidates.append(h)
    
    # Triangular / smooth patterns
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for shift in [0, 0.5, 1.0]:
            h = np.maximum(0.0, 1.0 + np.cos(2*np.pi*freq*X + shift*np.pi))
            if np.sum(h) > 0:
                candidates.append(h)
    
    # Gaussian-like bumps
    for center in np.linspace(-0.2, 0.2, 5):
        for width in [0.05, 0.1, 0.15, 0.2]:
            h = np.exp(-((X - center)**2) / (2*width**2))
            candidates.append(h)
    
    # Random initializations
    np.random.seed(42)
    for _ in range(30):
        h = np.random.exponential(1.0, size=N)
        candidates.append(h)
    
    # Sort by initial score
    scored_candidates = [(score(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Optimize from top candidates
    top_k = 15
    for init_score, init_h in scored_candidates[:top_k]:
        if time.time() > deadline - 20:
            break
        
        h0 = init_h.copy()
        # Normalize to avoid scale issues
        if np.sum(h0) > 0:
            h0 = h0 / np.sum(h0)
        
        def neg_obj(h):
            return -score(h)
        
        # Use SLSQP with bounds
        bounds = [(0.0, None)] * N
        
        try:
            res = minimize(neg_obj, h0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.success or res.fun < -init_score:
                h_opt = np.maximum(res.x, 0.0)
                s = score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # High-precision polish pass
    if time.time() < deadline - 10:
        h0 = best_h.copy()
        if np.sum(h0) > 0:
            h0 = h0 / np.sum(h0)
        
        def neg_obj(h):
            return -score(h)
        
        bounds = [(0.0, None)] * N
        
        try:
            res = minimize(neg_obj, h0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-14})
            if res.fun < -best_score:
                h_opt = np.maximum(res.x, 0.0)
                s = score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Block-coordinate refinement
    if time.time() < deadline - 5:
        h = best_h.copy()
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(N):
                h_old = h.copy()
                # Try varying h[i]
                best_val = score(h_old)
                best_h_i = h_old[i]
                # Coarse search
                for scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]:
                    h_test = h_old.copy()
                    h_test[i] = h_old[i] * scale
                    s = score(h_test)
                    if s > best_val:
                        best_val = s
                        best_h_i = h_test[i]
                # Fine search around best
                if best_h_i > 0:
                    for delta in np.linspace(-0.2*best_h_i, 0.2*best_h_i, 20):
                        h_test = h_old.copy()
                        h_test[i] = max(0.0, best_h_i + delta)
                        s = score(h_test)
                        if s > best_val:
                            best_val = s
                            best_h_i = h_test[i]
                h[i] = best_h_i
                if best_val > score(h_old) + 1e-12:
                    improved = True
        
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    return list(best_h)
# EVOLVE-BLOCK-END
