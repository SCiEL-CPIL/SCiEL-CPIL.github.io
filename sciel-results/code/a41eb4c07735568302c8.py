import numpy as np
from scipy.optimize import minimize
import time

N = 100
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

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
    # h is already in [0,1] from L-BFGS-B bounds
    # Sum constraint handled via penalty
    s = np.sum(h)
    target = N / 2.0
    penalty = 1e6 * (s - target) ** 2
    return _soft_max_overlap(h, beta) + penalty

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return np.full(N, 0.5)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Ensure sum constraint for seed
    best_h = np.full(N, 0.5)
    
    bounds = [(0.0, 1.0)] * N
    
    # Generate diverse initial points
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Block patterns
    for block in [N//4, N//3, N//5, 10, 20, 25, 50]:
        h = np.zeros(N)
        for start in range(0, N, block):
            h[start:start+block] = 1.0
        # Adjust to satisfy sum = N/2
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
    for _ in range(10):
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
    
    # Optimize each initial point
    for i, h0 in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Ensure exact sum
        s = np.sum(h0)
        if s > 0:
            h0 = h0 * (N/2) / s
        h0 = np.clip(h0, 0.001, 0.999)
        # Fine-tune sum
        diff = N/2 - np.sum(h0)
        h0 += diff / N
        h0 = np.clip(h0, 0.0, 1.0)
        
        try:
            res = minimize(
                _objective, h0,
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
            # Distribute correction
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
    if time.time() - START_TIME < TIME_LIMIT - 15:
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
