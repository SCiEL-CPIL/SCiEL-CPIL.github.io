import numpy as np
import time

N = 100
HALF = N // 2
DEADLINE = 110.0

def _compute_C(h):
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    return autocorr

def _max_overlap_from_C(C):
    return HALF - np.min(C)

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    best_val = float('inf')
    best_h = np.full(N, 0.5)
    
    # Start with constant vector
    h = np.full(N, 0.5)
    C = _compute_C(h)
    cur_val = _max_overlap_from_C(C)
    best_val = cur_val
    best_h = h.copy()
    
    # Gradient ascent on min(C)
    # We want to maximize min(C).
    # Let f(h) = min(C(h)).
    # Gradient of C_k is h_{i-k} + h_{i+k}.
    # We can use a simple hill climbing / gradient step.
    
    eta = 0.01
    for _ in range(5000):
        if time.time() - t0 > DEADLINE:
            break
            
        C = _compute_C(h)
        cur_val = _max_overlap_from_C(C)
        
        if cur_val < best_val:
            best_val = cur_val
            best_h = h.copy()
            
        # Find the shift k with minimum C
        k_min = np.argmin(C)
        
        # Compute gradient of C_k_min w.r.t h
        # grad[i] = h[(i-k_min)%N] + h[(i+k_min)%N]
        grad = h[(np.arange(N) - k_min) % N] + h[(np.arange(N) + k_min) % N]
        
        # Update h
        h_new = h + eta * grad
        
        # Project to sum constraint
        # We need sum(h_new) = N/2
        # Shift
        shift = (HALF - np.sum(h_new)) / N
        h_new = h_new + shift
        
        # Project to bounds [0, 1]
        # Simple clipping might break sum, so we do iterative clipping
        for _ in range(10):
            h_new = np.clip(h_new, 0, 1)
            shift = (HALF - np.sum(h_new)) / N
            h_new = h_new + shift
            # If sum is close, break
            if abs(np.sum(h_new) - HALF) < 1e-6:
                break
            # If clipping causes issues, we might need a more robust projection
            # But for small eta, it should be fine.
            
        h = h_new
        
    return best_h
