import numpy as np
import time

N = 50

def score_vector(h):
    # h is a numpy array of length N
    l1 = np.sum(h)
    if l1 == 0:
        return 0
    l2_sq = np.sum(h**2)
    # Compute autoconvolution
    # Using FFT for speed, or direct if N is small
    # N=50 is small, direct is fast enough
    # But we need to be careful with the definition of convolution
    # "autoconvolution" usually means linear convolution
    # The prompt mentions "supremum norm of the autoconvolution"
    # We should check if it's circular or linear.
    # "heights on equal intervals of [-1/4, 1/4]" suggests a finite support.
    # Linear convolution is standard for "autoconvolution" of a signal with finite support.
    
    # Linear convolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(conv))
    
    return l2_sq / (l1 * max_conv)

def construct():
    start_time = time.time()
    best_score = 0
    best_h = np.ones(N) # Fallback to seed
    
    # Try different sparsity levels
    # The seed is dense (K=50).
    # We suspect sparse is better.
    
    for k in range(1, N + 1):
        # Try a few random configurations for this k
        for trial in range(100):
            if time.time() - start_time > 110:
                break
            
            # Random indices
            idx = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            
            # Local improvement: try to swap a 1 with a 0
            # Or just evaluate
            s = score_vector(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
                
        # Greedy construction for this k
        h = np.zeros(N)
        for i in range(k):
            best_pos = -1
            best_val = -1
            for pos in range(N):
                if h[pos] == 0:
                    h[pos] = 1
                    s = score_vector(h)
                    if s > best_val:
                        best_val = s
                        best_pos = pos
                    h[pos] = 0
            if best_pos != -1:
                h[best_pos] = 1
            else:
                break
            
            s = score_vector(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
                
    # Convert to list of floats
    return list(best_h)
