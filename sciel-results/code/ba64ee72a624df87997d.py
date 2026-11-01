import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n_fft = 1 << (2 * N - 1).bit_length()
    
    def calc_psl(seq):
        F = np.fft.rfft(seq, n=n_fft)
        R = np.fft.irfft(F * np.conj(F), n=n_fft)
        peak = R[0]
        if peak == 0:
            return 0.0
        sidelobes = R[1:N]
        max_side = np.max(np.abs(sidelobes))
        return max_side / peak

    best_h = None
    best_val = np.inf
    
    # Try a few random starts to find a good initial basin
    for _ in range(10):
        h = np.random.choice([-1, 1], size=N)
        val = calc_psl(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    if best_h is None:
        return np.ones(N)
        
    h = best_h
    current_val = best_val
    
    T = 0.5
    T_min = 0.001
    alpha = 0.9999
    
    while time.time() < deadline:
        i = np.random.randint(N)
        old_val = h[i]
        h[i] = -old_val
        
        new_val = calc_psl(h)
        
        delta = new_val - current_val
        
        if delta < 0:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            if T > T_min:
                p = np.exp(-delta / T)
                if np.random.random() < p:
                    current_val = new_val
                else:
                    h[i] = old_val
            else:
                h[i] = old_val
        
        T *= alpha

    return best_h
