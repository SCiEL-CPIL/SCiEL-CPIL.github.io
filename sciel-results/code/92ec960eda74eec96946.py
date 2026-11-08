import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute constants
    w = 1.0 / (N + 1)  # spacing for y on [-0.5, 0.5] with m+2 points where m = 2*N-1

    def compute_value(h):
        """Compute the objective value for given heights h."""
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
            return -np.inf
        h = np.maximum(h, 0.0)
        
        g = np.convolve(h, h)
        m = len(g)  # should be 2*N - 1 = 99
        y = np.zeros(m + 2)
        y[1:m+1] = g
        
        # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
        # Note: the sum is over i from 0 to m, so there are m+1 terms
        terms = y[:m+1]**2 + y[:m+1] * y[1:m+2] + y[1:m+2]**2
        l2 = (w / 3.0) * np.sum(terms)
        
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        norm_inf = np.max(np.abs(g))
        
        if norm_1 == 0 or norm_inf == 0:
            return 0.0
        
        return l2 / (norm_1 * norm_inf)

    def objective(h):
        """Negative of value for minimization."""
        val = compute_value(h)
        if not np.isfinite(val):
            return 1e10
        return -val

    # Seed value
    seed_h = np.ones(N)
    seed_val = compute_value(seed_h)

    best_h = seed_h.copy()
    best_val = seed_val

    # Generate diverse starting points
    def generate_start(kind):
        if kind == 0:
            return np.ones(N)
        elif kind == 1:
            # Power law concentrated in center
            idx = np.arange(N)
            center = (N - 1) / 2.0
            h = np.exp(-((idx - center) ** 2) / (N * 0.5))
            return h / np.max(h) if np.max(h) > 0 else h
        elif kind == 2:
            # Sparse spike in center
            h = np.zeros(N)
            center = N // 2
            for d in range(5):
                if center - d >= 0:
                    h[center - d] = 1.0 - 0.15 * d
                if center + d < N:
                    h[center + d] = 1.0 - 0.15 * d
            return h
        elif kind == 3:
            # Two spikes
            h = np.zeros(N)
            c1 = N // 4
            c2 = 3 * N // 4
            for d in range(4):
                if c1 - d >= 0:
                    h[c1 - d] = 1.0 - 0.2 * d
                if c1 + d < N:
                    h[c1 + d] = 1.0 - 0.2 * d
                if c2 - d >= 0:
                    h[c2 - d] = 1.0 - 0.2 * d
                if c2 + d < N:
                    h[c2 + d] = 1.0 - 0.2 * d
            return h
        elif kind == 4:
            # Random smooth
            h = np.random.rand(N)
            # Smooth it
            kernel = np.array([0.2, 0.6, 0.2])
            h = np.convolve(h, kernel, mode='same')
            h = np.maximum(h, 0)
            return h / np.max(h) if np.max(h) > 0 else h
        elif kind == 5:
            # Cosine bump
            idx = np.linspace(0, np.pi, N)
            h = np.cos(idx - np.pi/2) ** 2
            h = np.maximum(h, 0)
            return h
        elif kind == 6:
            # Triangular
            idx = np.arange(N)
            center = (N - 1) / 2.0
            h = np.maximum(0, 1.0 - np.abs(idx - center) / (N * 0.4))
            return h
        else:
            return np.ones(N)

    # Try multiple starts with Nelder-Mead
    starts = []
    for kind in range(7):
        starts.append(generate_start(kind))
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(3):
        h = np.random.rand(N)
        h = np.maximum(h, 0)
        if np.max(h) > 0:
            h = h / np.max(h)
        starts.append(h)

    for h0 in starts:
        if time.time() > deadline:
            break
        
        h0 = np.asarray(h0, dtype=float).copy()
        h0 = np.maximum(h0, 0.0)
        
        try:
            result = minimize(
                objective,
                h0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            h_opt = np.maximum(result.x, 0.0)
            val = compute_value(h_opt)
            
            if np.isfinite(val) and val > best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    # Final polish on best
    if time.time() < deadline:
        try:
            result = minimize(
                objective,
                best_h.copy(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 10000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            
            h_opt = np.maximum(result.x, 0.0)
            val = compute_value(h_opt)
            
            if np.isfinite(val) and val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
# EVOLVE-BLOCK-END
