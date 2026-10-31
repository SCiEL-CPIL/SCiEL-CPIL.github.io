import numpy as np
import time

N = 100

def compute_score(h):
    # h is a numpy array of length N
    # Calculate circular convolution of h with h
    # Overlap of h with 1-h is 1 - (h*h)
    # We want to minimize the max of this.
    
    # Using FFT for speed
    H = np.fft.fft(h)
    C = np.fft.ifft(H * np.conj(H)).real
    # C[k] is the circular autocorrelation
    # The overlap integral is roughly (1/N) * sum(h * (1-h_shifted))
    # sum(h * (1-h_shifted)) = sum(h) - sum(h*h_shifted) = N*0.5 - C[k]
    # So value[k] = 0.5 - C[k]/N
    # We want to minimize max(value[k])
    # This is equivalent to minimizing max(0.5 - C[k]/N)
    # Which is equivalent to maximizing min(C[k]/N)
    
    min_c = np.min(C) / N
    max_overlap = 0.5 - min_c
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s
    
    # Start with the seed
    best_h = np.full(N, 0.5)
    best_score = compute_score(best_h)
    
    # Try to improve using random restarts and local search
    # Since the seed is likely optimal, this might not find anything better.
    # But we must try.
    
    # Simple gradient descent on the smooth surrogate
    # Objective: minimize max(0.5 - C[k]/N)
    # Surrogate: minimize sum(exp(alpha * (0.5 - C[k]/N)))
    
    alpha = 10.0
    lr = 0.01
    
    h = best_h.copy()
    
    for i in range(1000):
        if time.time() > deadline:
            break
            
        # Compute gradient of the surrogate
        C = np.fft.ifft(np.fft.fft(h) * np.conj(np.fft.fft(h))).real
        vals = 0.5 - C / N
        # Soft max
        max_val = np.max(vals)
        exp_vals = np.exp(alpha * (vals - max_val))
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of C[k] w.r.t h
        # C[k] = sum h[j] h[j-k]
        # dC[k]/dh[i] = h[i-k] + h[i+k]
        # So d(vals[k])/dh[i] = -(h[i-k] + h[i+k])/N
        
        # We want to decrease vals, so we update h in the direction of -grad(vals)
        # grad_total[i] = sum_k weights[k] * (h[i-k] + h[i+k])/N
        
        # This is a convolution of weights with h
        W = np.fft.fft(weights)
        H = np.fft.fft(h)
        grad = np.fft.ifft(W * H + W * np.conj(H)).real / N # Approximation
        
        # Update
        h_new = h - lr * grad
        # Project to [0,1] and sum N/2
        h_new = np.clip(h_new, 0, 1)
        # Adjust sum
        diff = N/2 - np.sum(h_new)
        h_new += diff / N
        h_new = np.clip(h_new, 0, 1)
        
        score_new = compute_score(h_new)
        
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
        
        h = h_new
        
    return best_h
