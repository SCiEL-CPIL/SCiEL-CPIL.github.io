import numpy as np
import time
from scipy.optimize import minimize

N = 50

def score_vector(h):
    # h is a numpy array of length N
    l1 = np.sum(h)
    if l1 == 0:
        return 0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(conv))
    if max_conv == 0:
        return 0
    return l2_sq / (l1 * max_conv)

def neg_score(h):
    return -score_vector(h)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    best_h = np.ones(N)
    best_score = score_vector(best_h)
    
    # Strategy: coordinate descent with continuous heights
    # Start from various initializations and optimize
    
    np.random.seed(42)
    
    # Try different initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) * 0.1)
    
    # 2. Sparse - single peak
    for pos in [0, N//2, N-1]:
        h = np.zeros(N)
        h[pos] = 1.0
        inits.append(h)
    
    # 3. Sparse - few peaks
    for k in [2, 3, 5, 10]:
        for _ in range(5):
            idx = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            inits.append(h)
    
    # 4. Random continuous
    for _ in range(10):
        h = np.random.exponential(1.0, size=N)
        inits.append(h)
    
    # 5. Symmetric patterns
    for k in [1, 2, 3, 5]:
        h = np.zeros(N)
        center = N // 2
        for i in range(k):
            if center - i >= 0:
                h[center - i] = 1.0
            if center + i < N:
                h[center + i] = 1.0
        inits.append(h)
    
    # Coordinate descent optimization
    for h0 in inits:
        if time.time() > deadline:
            break
        
        # Use L-BFGS-B with non-negativity constraint
        bounds = [(0, None)] * N
        
        try:
            result = minimize(
                neg_score, h0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_h = result.x.copy()
        except Exception:
            pass
        
        # Also try coordinate descent manually for robustness
        h = h0.copy()
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                old_val = h[i]
                # Try a few values
                candidates = [old_val * 0.5, old_val * 1.5, old_val + 0.1, old_val - 0.1, 0.0, 1.0]
                candidates = [max(0, c) for c in candidates]
                best_cand = old_val
                best_s = score_vector(h)
                for c in candidates:
                    h[i] = c
                    s = score_vector(h)
                    if s > best_s:
                        best_s = s
                        best_cand = c
                h[i] = best_cand
                if best_cand != old_val:
                    improved = True
            
            s = score_vector(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
            
            if not improved:
                break
    
    return list(best_h)
