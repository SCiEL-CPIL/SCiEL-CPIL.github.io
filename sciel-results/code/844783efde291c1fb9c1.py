import numpy as np
from scipy.optimize import minimize
import time

N = 100
START_TIME = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill

def _compute_overlaps(h):
    """Compute overlap at all discrete shifts. h is array of N step heights."""
    n = len(h)
    overlaps = np.empty(2 * n - 1)
    for k in range(-n + 1, n):
        # k is shift in step units
        total = 0.0
        for i in range(n):
            j = i + k
            if 0 <= j < n:
                total += h[i] * (1.0 - h[j])
            else:
                total += h[i]
        overlaps[k + n - 1] = total * (2.0 / n)
    return overlaps

def _max_overlap(h):
    """Hard max of overlaps."""
    return np.max(_compute_overlaps(h))

def _soft_max_overlap(h, beta=50.0):
    """Log-sum-exp surrogate for max overlap."""
    ov = _compute_overlaps(h)
    m = np.max(ov)
    if m < 1e-12:
        return m
    return m + np.log(np.sum(np.exp(beta * (ov - m)))) / beta

def _objective(h, beta=50.0):
    """Objective with soft constraint on sum and bounds."""
    s = np.sum(h)
    target = N / 2.0
    penalty = 1e6 * (s - target) ** 2
    return _soft_max_overlap(h, beta) + penalty

def _project_to_simplex(h, target_sum):
    """Project h onto the hyperplane sum(h) = target_sum, then clip to [0,1] and re-adjust sum."""
    # Simple projection: shift to match sum, then clip, then shift again if needed
    h = h.copy()
    diff = target_sum - np.sum(h)
    h += diff / len(h)
    h = np.clip(h, 0.0, 1.0)
    # If clipping changed the sum, adjust again
    diff = target_sum - np.sum(h)
    # Only adjust elements that are not at bounds to preserve bounds if possible?
    # For simplicity, just add uniform adjustment, might violate bounds slightly, 
    # but L-BFGS-B will handle bounds. For subgradient, we want strict feasibility.
    # A better projection would be iterative, but this is fast.
    h += diff / len(h)
    h = np.clip(h, 0.0, 1.0)
    return h

def _subgradient_ascent(h, steps=200, lr_init=0.05):
    """
    Maximize min_k autocorrelation C_k(h) = sum h_i h_{i+k}.
    This is equivalent to minimizing max_k overlap O_k(h) = N/2 - C_k(h).
    """
    h = h.copy()
    h = _project_to_simplex(h, N/2)
    
    for step in range(steps):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Compute autocorrelations C_k
        # C_k = sum_i h_i h_{i+k}
        c_vals = np.zeros(2*N - 1)
        for k in range(-N + 1, N):
            total = 0.0
            for i in range(N):
                j = i + k
                if 0 <= j < N:
                    total += h[i] * h[j]
            c_vals[k + N - 1] = total
        
        min_c = np.min(c_vals)
        # Active indices: those close to the minimum
        threshold = min_c + 1e-4
        active = np.where(c_vals <= threshold)[0]
        
        if len(active) == 0:
            active = np.array([np.argmin(c_vals)])
            
        # Compute subgradient of min(C)
        # grad C_k at h_j = h_{j+k} + h_{j-k} (with 0 for OOB)
        g = np.zeros(N)
        for idx in active:
            k = idx - (N - 1)
            for j in range(N):
                val = 0.0
                if 0 <= j + k < N:
                    val += h[j + k]
                if 0 <= j - k < N:
                    val += h[j - k]
                g[j] += val / len(active)
                
        # Learning rate decay
        lr = lr_init / (1 + 0.01 * step)
        
        # Ascent step: h = h + lr * g
        h = h + lr * g
        
        # Project to feasible set
        h = _project_to_simplex(h, N/2)
        
    return h

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return np.full(N, 0.5)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    bounds = [(0.0, 1.0)] * N
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    inits = []
    
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Block patterns
    for block in [N//4, N//3, N//5, 10, 20, 25, 50]:
        h = np.zeros(N)
        for start in range(0, N, block):
            h[start:start+block] = 1.0
        s = np.sum(h)
        if s > 0:
            h *= (N/2) / s
            h = np.clip(h, 0, 1)
        inits.append(h.copy())
        
    # 3. Sine patterns
    for freq in [1, 2, 3, 5, 8]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h = 0.5 + 0.4*np.sin(t)
        h = np.clip(h, 0, 1)
        s = np.sum(h)
        if s > 0:
            h *= (N/2) / s
            h = np.clip(h, 0, 1)
        inits.append(h.copy())
        
    # 4. Random
    for _ in range(15):
        h = rng.random(N)
        s = np.sum(h)
        if s > 0:
            h *= (N/2) / s
            h = np.clip(h, 0, 1)
        inits.append(h.copy())
        
    # 5. Alternating
    h = np.tile([0.2, 0.8], N//2)
    if len(h) < N:
        h = np.append(h, 0.5)
    h = h[:N]
    s = np.sum(h)
    if s > 0:
        h *= (N/2) / s
        h = np.clip(h, 0, 1)
    inits.append(h.copy())
    
    # 6. Ramp
    h = np.linspace(0.1, 0.9, N)
    s = np.sum(h)
    if s > 0:
        h *= (N/2) / s
        h = np.clip(h, 0, 1)
    inits.append(h.copy())
        
    # 7. Two-level: some 0s, some 1s, adjusted
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        h = np.zeros(N)
        idx = rng.choice(N, size=int(N*frac), replace=False)
        h[idx] = 1.0
        s = np.sum(h)
        if s > 0:
            h *= (N/2) / s
            h = np.clip(h, 0, 1)
        inits.append(h.copy())
        
    # Phase 1: Subgradient Ascent for each init
    for i, h0 in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        # Ensure exact sum
        s = np.sum(h0)
        if s > 0:
            h0 = h0 * (N/2) / s
        h0 = np.clip(h0, 0.0, 1.0)
        diff = N/2 - np.sum(h0)
        h0 += diff / N
        h0 = np.clip(h0, 0.0, 1.0)
        
        try:
            # Run subgradient ascent
            h_sub = _subgradient_ascent(h0, steps=100, lr_init=0.05)
            
            # Polish with L-BFGS-B
            res = minimize(
                _objective, h_sub,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = res.x
            # Project to satisfy sum exactly
            s = np.sum(h_opt)
            if s > 0:
                h_opt = h_opt * (N/2) / s
            h_opt = np.clip(h_opt, 0.0, 1.0)
            
            # Polish: fine-tune to get exact sum
            diff = N/2 - np.sum(h_opt)
            if abs(diff) > 1e-12:
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0.0, 1.0)
            
            score = _max_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass

    # Multi-scale beta polish on best
    if time.time() - START_TIME < TIME_LIMIT - 10:
        for beta in [100, 200, 500]:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            try:
                def obj_polish(h, b=beta):
                    ov = _compute_overlaps(h)
                    m = np.max(ov)
                    penalty = 1e6 * (np.sum(h) - N/2) ** 2
                    return m + np.log(np.sum(np.exp(b*(ov-m))))/b + penalty
                
                res = minimize(
                    obj_polish, best_h,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = res.x
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt * (N/2) / s
                h_opt = np.clip(h_opt, 0.0, 1.0)
                diff = N/2 - np.sum(h_opt)
                if abs(diff) > 1e-12:
                    h_opt += diff / N
                    h_opt = np.clip(h_opt, 0.0, 1.0)
                
                score = _max_overlap(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final: ensure exact sum
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h * (N/2) / s
    best_h = np.clip(best_h, 0.0, 1.0)
    # Exact sum fix
    diff = N/2 - np.sum(best_h)
    if abs(diff) > 1e-15:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
