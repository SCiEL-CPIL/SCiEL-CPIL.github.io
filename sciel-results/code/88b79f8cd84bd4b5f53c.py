import numpy as np
import time

N = 400

def _compute_all_autocorr(h):
    n = len(h)
    A = np.zeros(n)
    for k in range(n):
        A[k] = np.dot(h[:n-k], h[k:])
    return A

def _thue_morse(n):
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        t = i
        count = 0
        while t & 1:
            count += 1
            t >>= 1
        seq[i] = 1 if count % 2 == 0 else -1
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try multiple initial sequences
    inits = [
        _thue_morse(N),
        np.ones(N, dtype=np.int8),
        np.random.choice([-1, 1], size=N),
    ]
    
    best_overall_val = float('inf')
    best_overall_h = None
    
    for h_init in inits:
        if time.time() > deadline:
            break
        h = h_init.copy().astype(np.int8)
        A = _compute_all_autocorr(h)
        best_val = float(np.max(np.abs(A[1:])))
        best_h = h.copy()
        
        # Local search
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                v = h[i]
                delta = np.zeros(N, dtype=np.float64)
                if i + 1 < N:
                    delta[1:N-i] += -2.0 * v * h[i+1:N].astype(np.float64)
                if i >= 1:
                    delta[1:i+1] += -2.0 * v * h[i-1::-1].astype(np.float64)
                
                A_new = A + delta
                new_max = float(np.max(np.abs(A_new[1:])))
                if new_max < best_val:
                    best_val = new_max
                    best_h = h.copy()
                    h[i] = -v
                    A = A_new
                    improved = True
        
        if best_val < best_overall_val:
            best_overall_val = best_val
            best_overall_h = best_h
    
    if best_overall_h is None:
        return _thue_morse(N)
    
    return best_overall_h
