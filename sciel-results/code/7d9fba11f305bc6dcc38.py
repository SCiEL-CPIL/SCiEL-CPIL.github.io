"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy import optimize

N = 50

def _score(h):
    """Compute the objective: (sum h^2) / (sum h * max autoconv)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return -np.inf
    h = np.maximum(h, 0.0)
    s2 = np.sum(h**2)
    s1 = np.sum(h)
    if s1 < 1e-15 or s2 < 1e-15:
        return 0.0
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(conv))
    if max_conv < 1e-15:
        return 0.0
    return s2 / (s1 * max_conv)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _score(best_h)
    
    # 1. Generate diverse initial patterns
    patterns = []
    
    # Single spike
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        patterns.append(h)
        
    # Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            patterns.append(h)
            
    # Random sparse
    np.random.seed(42)
    for _ in range(20):
        h = np.zeros(N)
        k = np.random.randint(1, 10)
        idx = np.random.choice(N, k, replace=False)
        h[idx] = 1.0
        patterns.append(h)
        
    # Random dense
    for _ in range(20):
        h = np.random.exponential(1.0, size=N)
        patterns.append(h)
        
    # Structured
    patterns.append(np.ones(N))
    patterns.append(np.linspace(0, 1, N))
    patterns.append(np.cos(np.pi * np.linspace(-0.5, 0.5, N))**2)
    
    # Evaluate patterns
    for h in patterns:
        if time.time() > deadline:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # 2. Global Optimization using Differential Evolution
    if time.time() < deadline - 20:
        def neg_score(h):
            h = np.maximum(h, 0.0)
            s = _score(h)
            return -s if np.isfinite(s) else 1e10
            
        bounds = [(0.0, 10.0) for _ in range(N)]
        
        try:
            # Use a smaller population for speed
            res = optimize.differential_evolution(
                neg_score, bounds, 
                seed=42, 
                maxiter=50, 
                tol=1e-4,
                popsize=10,
                polish=False
            )
            if time.time() < deadline - 10:
                h_opt = np.maximum(res.x, 0.0)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
            
    # 3. Local Refinement using Nelder-Mead
    if time.time() < deadline - 5:
        def neg_score(h):
            h = np.maximum(h, 0.0)
            s = _score(h)
            return -s if np.isfinite(s) else 1e10
            
        try:
            res = optimize.minimize(
                neg_score, best_h, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            h_opt = np.maximum(res.x, 0.0)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
            
    # 4. Final check and cleanup
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
# EVOLVE-BLOCK-END
