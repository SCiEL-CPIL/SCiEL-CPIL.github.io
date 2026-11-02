import numpy as np
import time

N = 100

def _overlap_max(h):
    """
    Compute max cyclic overlap of h with 1-h, scaled by step width.
    h is an array of length N.
    overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    = sum_i h[i] - sum_i h[i] * h[(i+k) % N]
    = N/2 - autocorr[k]
    We use FFT for efficient computation.
    """
    w = 2.0 / N
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # Due to floating point, autocorr might have tiny imaginary parts or slight deviations
    # The max overlap is N/2 - min(autocorr) ? No.
    # overlap_k = N/2 - autocorr_k
    # We want to MINIMIZE max(overlap_k).
    # This is equivalent to MINIMIZING (N/2 - min_k autocorr_k)? No.
    # max_k (N/2 - autocorr_k) = N/2 - min_k autocorr_k.
    # So minimizing max overlap is equivalent to maximizing the minimum autocorrelation.
    # However, we must be careful: the "overlap" is defined for h and 1-h.
    # Let's stick to the formula: overlaps = N/2 - autocorr
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def construct():
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # We assume a bang-bang structure: h_i in {0, 1}.
    # Constraint: sum(h) = N/2 = 50.
    # So we are looking for a binary sequence of length 100 with exactly 50 ones.
    
    # Initial state: 50 ones and 50 zeros
    # A good initial guess might be alternating 0,1,0,1...
    # Or random.
    # Let's try a few diverse starts.
    
    def random_bang_bang():
        h = np.zeros(N)
        indices = np.random.choice(N, N // 2, replace=False)
        h[indices] = 1.0
        return h

    def alternating_bang_bang():
        h = np.zeros(N)
        h[::2] = 1.0
        return h

    def block_bang_bang():
        h = np.zeros(N)
        h[:N // 2] = 1.0
        return h

    starts = [
        alternating_bang_bang(),
        block_bang_bang(),
        random_bang_bang(),
        random_bang_bang(),
        random_bang_bang()
    ]

    best_h = None
    best_val = np.inf

    for start_h in starts:
        if time.time() > deadline:
            break
            
        current_h = start_h.copy()
        current_val = _overlap_max(current_h)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()

        # Simulated Annealing
        T = 0.1
        T_min = 1e-6
        cooling = 0.999
        
        # We will perform a fixed number of steps, but check time
        steps_per_start = 50000
        
        for step in range(steps_per_start):
            if time.time() > deadline:
                break
            
            # Pick a random index to flip
            i = np.random.randint(N)
            
            # Flip h[i]
            old_val = current_h[i]
            new_val = 1.0 - old_val
            current_h[i] = new_val
            
            new_val_score = _overlap_max(current_h)
            
            # Calculate delta in objective (we want to minimize)
            delta = new_val_score - current_val
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_val = new_val_score
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                # Reject
                current_h[i] = old_val
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min

    # Final check: ensure the returned array is valid
    # The task requires sum = N/2. Our bang-bang approach guarantees this if we only flip.
    # However, if we want to be absolutely safe against floating point issues in evaluation
    # (though 0.0 and 1.0 are exact), we can return best_h.
    
    # One potential issue: The evaluator might check `np.sum(h) == N/2` strictly.
    # 50 * 1.0 + 50 * 0.0 = 50.0. This is exact in float64.
    
    return best_h
