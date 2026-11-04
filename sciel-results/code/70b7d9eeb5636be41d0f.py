import numpy as np
import time

N = 400

def construct():
    # Initialize with Thue-Morse sequence
    t = np.zeros(N, dtype=int)
    t[0] = 1
    for i in range(1, N):
        if i % 2 == 0:
            t[i] = t[i // 2]
        else:
            t[i] = -t[i // 2]
    
    f = t.astype(float)
    
    def get_peak(f):
        F = np.fft.fft(f)
        C = np.fft.ifft(F**3)
        return np.max(np.abs(C))
    
    current_peak = get_peak(f)
    
    start_time = time.time()
    deadline = 110.0
    
    # Local search
    # Random restarts with hill climbing
    best_f = f.copy()
    best_peak = current_peak
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Start from best found so far or random
        if restart == 0:
            f = best_f.copy()
            current_peak = best_peak
        else:
            f = np.random.choice([-1.0, 1.0], size=N)
            current_peak = get_peak(f)
            if current_peak < best_peak:
                best_f = f.copy()
                best_peak = current_peak
        
        # Hill climbing
        for step in range(2000):
            if time.time() > deadline:
                break
                
            idx = np.random.randint(N)
            f[idx] *= -1
            
            new_peak = get_peak(f)
            
            if new_peak < current_peak:
                current_peak = new_peak
                if current_peak < best_peak:
                    best_f = f.copy()
                    best_peak = current_peak
            else:
                f[idx] *= -1
        
        if time.time() > deadline:
            break
            
    return best_f
