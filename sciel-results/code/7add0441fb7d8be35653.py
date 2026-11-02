import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial state: all 0.5
    h = np.full(N, 0.5)
    
    def get_max_overlap(h):
        # Compute cyclic cross-correlation of h and 1-h
        b = 1.0 - h
        fft_h = np.fft.fft(h)
        fft_b = np.fft.fft(b)
        # Correlation
        corr = np.real(np.fft.ifft(fft_h * np.conj(fft_b)))
        return np.max(corr)

    current_overlap = get_max_overlap(h)
    best_h = h.copy()
    best_overlap = current_overlap
    
    # Simulated Annealing
    T = 0.5
    T_min = 1e-4
    cooling_rate = 0.999
    
    # Pre-allocate arrays if possible? No, h changes.
    
    while time.time() < deadline:
        # Pick two distinct indices
        i, j = np.random.choice(N, 2, replace=False)
        
        # Random delta
        delta = np.random.uniform(-0.1, 0.1)
        
        new_h_i = h[i] + delta
        new_h_j = h[j] - delta
        
        if 0 <= new_h_i <= 1 and 0 <= new_h_j <= 1:
            h[i] = new_h_i
            h[j] = new_h_j
            
            new_overlap = get_max_overlap(h)
            
            delta_e = new_overlap - current_overlap
            
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                current_overlap = new_overlap
                if current_overlap < best_overlap:
                    best_overlap = current_overlap
                    best_h = h.copy()
            else:
                # Revert
                h[i] -= delta
                h[j] += delta
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_h
