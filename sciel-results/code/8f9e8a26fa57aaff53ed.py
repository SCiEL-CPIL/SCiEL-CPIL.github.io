import numpy as np
import time

N = 100

def autocorr_circular(h):
    """Compute circular autocorrelation: ac[k] = sum_i h[i]*h[(i+k)%N]"""
    H = np.fft.fft(h)
    power = H * np.conj(H)
    ac = np.fft.ifft(power).real
    return ac

def max_overlap(h):
    """Compute max_k sum_i h[i]*(1-h[(i+k)%N]) = 50 - min_k autocorr[k]"""
    ac = autocorr_circular(h)
    return 50.0 - np.min(ac)

def construct():
    start_time = time.time()
    deadline = 120.0
    
    best_h = np.full(N, 0.5)
    best_score = max_overlap(best_h)  # Should be 25.0
    
    np.random.seed(12345)
    
    # Hill climbing with random restarts
    for restart in range(50):
        if time.time() - start_time > deadline - 5:
            break
        
        # Initialize
        if restart == 0:
            h = best_h.copy()
        else:
            h = np.random.rand(N)
            h = np.clip(h, 0, 1)
            diff = 50.0 - np.sum(h)
            mask = (h > 0.05) & (h < 0.95)
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
        
        cur_score = max_overlap(h)
        
        # Hill climbing
        for step in range(5000):
            if time.time() - start_time > deadline - 3:
                break
            
            # Random perturbation
            idx = np.random.randint(N)
            delta = np.random.uniform(-0.05, 0.05)
            h[idx] = np.clip(h[idx] + delta, 0, 1)
            
            # Fix sum
            diff = 50.0 - np.sum(h)
            if abs(diff) > 1e-10:
                mask = (h > 0.001) & (h < 0.999)
                if np.sum(mask) > 0:
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0, 1)
            
            new_score = max_overlap(h)
            if new_score < cur_score:
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_h = h.copy()
            else:
                h[idx] = np.clip(h[idx] - delta, 0, 1)
                # Re-fix sum
                diff = 50.0 - np.sum(h)
                if abs(diff) > 1e-10:
                    mask = (h > 0.001) & (h < 0.999)
                    if np.sum(mask) > 0:
                        h[mask] += diff / np.sum(mask)
                        h = np.clip(h, 0, 1)
    
    # Ensure exact sum
    diff = 50.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
