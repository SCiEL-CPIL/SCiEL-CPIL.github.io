import numpy as np
import time
from scipy.optimize import minimize

N = 100

def compute_score(h):
    H = np.fft.fft(h)
    C = np.fft.ifft(H * np.conj(H)).real
    return float(np.max(N/2 - C))

def construct():
    start = time.time()
    deadline = start + 140
    
    np.random.seed(42)
    
    # Phase 1: Binary SA
    h = np.zeros(N)
    ones = np.random.choice(N, N//2, replace=False)
    h[ones] = 1.0
    
    current_score = compute_score(h)
    best_score = current_score
    best_h = h.copy()
    
    T = 5.0
    T_min = 0.005
    cooling = 0.9998
    
    while time.time() < deadline * 0.6 and T > T_min:
        pos1 = np.where(h == 1.0)[0]
        pos0 = np.where(h == 0.0)[0]
        if len(pos1) == 0 or len(pos0) == 0:
            break
        i = pos1[np.random.randint(len(pos1))]
        j = pos0[np.random.randint(len(pos0))]
        
        h[i] = 0.0
        h[j] = 1.0
        
        new_score = compute_score(h)
        delta = new_score - current_score
        
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            h[i] = 1.0
            h[j] = 0.0
        
        T *= cooling
    
    # Phase 2: Continuous refinement with SLSQP
    if time.time() < deadline - 20:
        try:
            def obj(x):
                return compute_score(x)
            
            res = minimize(
                obj,
                best_h.astype(float),
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}],
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if res.fun < best_score:
                best_score = float(res.fun)
                best_h = res.x
        except Exception:
            pass
    
    # Ensure validity
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N/2 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
