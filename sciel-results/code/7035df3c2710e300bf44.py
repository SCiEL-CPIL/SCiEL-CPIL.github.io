import numpy as np
import time

N = 50

def compute_score(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    if sup_c < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_c)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12):
            h = np.maximum(h, 0.0)
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s

    # 1. Single spike (Theoretical max candidate)
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    try_candidate(h_spike)
    
    # 2. Center spike
    h_center = np.zeros(N)
    h_center[N//2] = 1.0
    try_candidate(h_center)

    # 3. Two spikes (Sidon-like)
    h_two = np.zeros(N)
    h_two[0] = 1.0
    h_two[10] = 1.0
    try_candidate(h_two)
    
    # 4. Uniform
    h_uni = np.ones(N) / N
    try_candidate(h_uni)
    
    # 5. Rapid multiplicative update from a few diverse starts
    # Focus on refining shapes that might break the 1.0 barrier if it exists
    # or confirm 1.0 is the max.
    
    def fast_multiplicative_update(h_init, n_iter=50):
        h = np.asarray(h_init, dtype=float).copy()
        h = np.maximum(h, 1e-10)
        best_local = 0.0
        best_local_h = h.copy()
        
        for _ in range(n_iter):
            if time.time() > deadline - 5:
                break
            c = np.correlate(h, h, mode='full')
            c_max = np.max(c)
            if c_max < 1e-15:
                break
            
            # Soft max weights
            temp = 0.1 * c_max
            weights = np.exp((c - c_max) / temp)
            weights /= np.sum(weights)
            
            # Approximate pressure
            # Using FFT for speed if N were larger, but N=50 is small
            # Simple loop for pressure
            pressure = np.zeros(N)
            for m in range(2*N - 1):
                w = weights[m]
                if w < 1e-8:
                    continue
                L = m - (N - 1)
                if L == 0:
                    pressure += (2.0 * w / c_max) * h
                elif L > 0:
                    # h[i-L] and h[i+L]
                    for i in range(N):
                        contrib = 0.0
                        if i - L >= 0: contrib += h[i - L]
                        if i + L < N: contrib += h[i + L]
                        pressure[i] += w * contrib / c_max
                else:
                    Lp = -L
                    for i in range(N):
                        contrib = 0.0
                        if i + Lp < N: contrib += h[i + Lp]
                        if i - Lp >= 0: contrib += h[i - Lp]
                        pressure[i] += w * contrib / c_max
            
            l2_sq = np.sum(h**2)
            l1 = np.sum(h)
            
            log_deriv = 2.0 * h**2 / l2_sq - h / l1 - pressure
            eta = 0.5
            h = h * np.exp(eta * log_deriv)
            h = np.maximum(h, 1e-10)
            
            s = compute_score(h)
            if s > best_local:
                best_local = s
                best_local_h = h.copy()
                
        return best_local_h, best_local

    # Try a few structured inits
    inits = []
    # Ramp
    inits.append(np.linspace(0, 1, N))
    # Gaussian
    x = np.arange(N)
    inits.append(np.exp(-((x - N//2)**2) / 10))
    # Sparse random
    np.random.seed(1)
    h_rand = np.zeros(N)
    h_rand[np.random.choice(N, 5, replace=False)] = 1.0
    inits.append(h_rand)
    
    for h_init in inits:
        if time.time() > deadline - 10:
            break
        h_ref, s_ref = fast_multiplicative_update(h_init, n_iter=100)
        try_candidate(h_ref)

    return list(best_h)
