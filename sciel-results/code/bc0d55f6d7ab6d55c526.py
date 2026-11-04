import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initialize with random binary sequence
    # 50% density
    f = np.random.randint(0, 2, N).astype(float)
    
    # Calculate initial score
    # We assume the score is proportional to max(f*f) / (sum(f))^2
    # Let's minimize max(f*f) / (sum(f))^2
    
    def get_score(f):
        # Compute convolution
        # Use FFT for speed
        F = np.fft.fft(f, n=2*N)
        conv = np.fft.ifft(F * F).real
        # The convolution is circular if we don't pad, but we padded to 2N
        # The linear convolution is in conv[0:N] and conv[N:2N] is wrap-around?
        # Actually, if we pad to 2N, the linear conv is conv[0:N].
        # The max is over the linear conv.
        max_val = np.max(conv[:N])
        sum_val = np.sum(f)
        if sum_val == 0:
            return float('inf')
        return max_val / (sum_val * sum_val)
    
    current_score = get_score(f)
    best_f = f.copy()
    best_score = current_score
    
    # Local search: flip bits
    # To speed up, we can try flipping bits that are part of the peak correlation
    # But a simple random restart or greedy flip is safer
    
    # Greedy flip:
    # Try flipping each bit, see if it reduces the score.
    # This is O(N^2) per evaluation if we recompute conv.
    # N=600. N^2=360k. 600 flips = 216M ops. Might be too slow in Python.
    # Use FFT update?
    # Flipping bit j: f_j -> 1-f_j.
    # Delta f = (1-2f_j) at j.
    # Delta F = (1-2f_j) * exp(-i 2 pi j / 2N).
    # New F = F + Delta F.
    # New Conv = ifft(New F * New F).
    
    # We will do a limited number of iterations.
    max_iterations = 5000
    for i in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Pick a random index to flip
        j = np.random.randint(N)
        old_f_j = f[j]
        f[j] = 1.0 - old_f_j
        
        new_score = get_score(f)
        
        if new_score < best_score:
            best_score = new_score
            best_f = f.copy()
        else:
            # Revert
            f[j] = old_f_j
            
    return list(best_f)
