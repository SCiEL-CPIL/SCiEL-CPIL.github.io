import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initialize with a few blocks
    # Try different patterns and keep the best
    best_h = None
    best_score = float('inf')
    
    patterns = [
        lambda i: 1.0 if i % 4 < 2 else 0.0,  # 2 blocks of 2
        lambda i: 1.0 if i % 8 < 4 else 0.0,  # 1 block of 4
        lambda i: 1.0 if i % 2 == 0 else 0.0, # Sparse
        lambda i: 1.0,                         # Uniform
        lambda i: 1.0 if 100 < i < 200 or 400 < i < 500 else 0.0 # Two wide blocks
    ]
    
    for pat in patterns:
        h = np.array([pat(i) for i in range(N)], dtype=float)
        if np.sum(h) == 0:
            h = np.ones(N)
        h = h / np.sum(h)
        
        # Optimize
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Compute autoconvolution
            # Use FFT
            f_len = 2 * N
            h_fft = np.fft.fft(h, f_len)
            c = np.fft.ifft(h_fft * h_fft).real[:N] # First N points? 
            # Actually convolution of length N with N is length 2N-1.
            # We need the max over the whole convolution.
            c_full = np.fft.ifft(h_fft * h_fft).real[:2*N-1]
            
            max_val = np.max(c_full)
            
            # Score is max_val / sum(h) ? 
            # Wait, the metric is max(g) / ||f||_1.
            # ||f||_1 = sum(h) * dx.
            # max(g) = max(c) * dx^2.
            # Ratio = (max(c) * dx^2) / (sum(h) * dx) = max(c) * dx / sum(h).
            # Since dx is constant, minimizing max(c)/sum(h) is equivalent.
            # We normalized sum(h) to 1, so score is proportional to max(c).
            
            if best_score is None or max_val < best_score:
                best_score = max_val
                best_h = h.copy()
                
            # Find index of max
            idx_max = np.argmax(c_full)
            
            # Gradient of c[idx_max] w.r.t h
            # c_k = sum_i h_i h_{k-i}
            # d c_k / d h_j = h_{k-j}
            # So grad = h reversed/shifted
            # h is 0..N-1.
            # For k in 0..2N-2:
            # grad_j = h_{k-j} if 0 <= k-j < N else 0.
            
            # Efficiently compute grad for the specific idx_max
            # We need h_{idx_max - j}
            # This is a shifted version of h.
            
            # Create a zero array for grad
            grad = np.zeros(N)
            # h_{idx_max - j} is non-zero when 0 <= idx_max - j < N
            # => idx_max - N < j <= idx_max
            # j in [max(0, idx_max - N + 1), min(N-1, idx_max)]
            
            lo = max(0, idx_max - N + 1)
            hi = min(N - 1, idx_max)
            
            if lo <= hi:
                # j ranges from lo to hi
                # index in h is idx_max - j
                # range of idx_max - j is idx_max - hi to idx_max - lo
                # idx_max - hi is 0 (if hi=idx_max) or something
                # Let's just slice
                # h_idx = idx_max - np.arange(lo, hi+1)
                # grad[lo:hi+1] = h[h_idx]
                
                # Note: h is defined on 0..N-1.
                # c_full index k corresponds to shift k.
                # h_i is at position i.
                # Overlap of h_i and h_{k-i}.
                # Yes.
                
                indices = idx_max - np.arange(lo, hi + 1)
                grad[lo:hi + 1] = h[indices]
            
            # Update h
            # h = h - eta * grad
            # Choose eta
            eta = 0.1
            h_new = h - eta * grad
            h_new[h_new < 0] = 0
            
            if np.sum(h_new) == 0:
                break
            h_new = h_new / np.sum(h_new)
            h = h_new
            
        if time.time() > deadline:
            break
            
    # Return the best h found
    if best_h is None:
        return [1.0] * N
    return list(best_h)
