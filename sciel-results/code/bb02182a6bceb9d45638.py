import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h)."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1]."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, alpha=500.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_h = None
    best_score = float('inf')
    
    # Initial guess: uniform
    h_uniform = np.full(N, 0.5)
    best_h = h_uniform.copy()
    best_score = _score(best_h)
    
    # Search over periods p
    # p must be at least ceil(N/2) to allow sum N/2 with h_i <= 1
    # Actually, sum is N/2, max sum for period p is p. So p >= N/2 = 50.
    for p in range(50, N + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Define the base vector b of length p
        # We will optimize b such that the tiled/truncated h has sum N/2
        # and minimizes max overlap.
        
        # The tiled vector h is constructed from b.
        # h[i] = b[i % p]
        # We need sum(h) = N/2.
        # Let k = N // p, r = N % p.
        # sum(h) = k * sum(b) + sum(b[:r])
        # This sum depends on b non-linearly (piecewise linear).
        # Instead, we can optimize b with a penalty for the sum constraint on the full h.
        
        def obj_b(b):
            # Construct h
            h = np.tile(b, N // p + 1)[:N]
            # Calculate smooth max overlap
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=500.0)
            
            # Penalty for sum constraint
            s = np.sum(h)
            sum_penalty = 100.0 * (s - N/2.0)**2
            
            # Penalty for bounds (though we use bounds in optimizer)
            bound_penalty = 0.0
            
            return smax + sum_penalty + bound_penalty

        def con_b(b):
            # We might not need explicit constraints if penalty is strong,
            # but let's try to enforce sum more tightly if possible.
            # Actually, SLSQP handles equality constraints well.
            h = np.tile(b, N // p + 1)[:N]
            return np.sum(h) - N/2.0

        # Initial guess for b: uniform value that would give sum N/2
        # Approx sum per element is 0.5.
        b0 = np.full(p, 0.5)
        
        bounds = [(0.0, 1.0)] * p
        
        try:
            res = minimize(
                obj_b, 
                b0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[{'type': 'eq', 'fun': con_b}],
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                b_opt = res.x
                h_opt = np.tile(b_opt, N // p + 1)[:N]
                h_opt = _normalize(h_opt) # Ensure exact sum
                score = _score(h_opt)
                
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
                    
        except Exception as e:
            pass

    # If no periodic solution found or all failed, fallback to global optimization
    if best_score == float('inf'):
        best_h = np.full(N, 0.5)
        best_score = _score(best_h)

    # Final polish on the best found using full N variables
    # This helps if the periodic assumption was too restrictive or if p=N
    if time.time() < deadline - 5.0:
        def obj_full(h):
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=1000.0)
            s = np.sum(h)
            sum_penalty = 1000.0 * (s - N/2.0)**2
            return smax + sum_penalty

        def con_full(h):
            return np.sum(h) - N/2.0

        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(
                obj_full, 
                best_h, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[{'type': 'eq', 'fun': con_full}],
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            
            if res.success:
                h_opt = _normalize(res.x)
                score = _score(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass

    return best_h
