import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    # Initialize with all ones (the seed)
    x = np.ones(N, dtype=np.int8)
    
    def compute_score(x):
        # Sum S
        S = np.sum(x)
        if S == 0:
            return float('inf')
        
        # Autoconvolution
        # Use FFT for efficiency
        # Pad to 2*N to avoid circular convolution
        n_fft = 2 * N
        X = np.fft.rfft(x, n=n_fft)
        C = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
        
        # Exclude lag 0
        C[0] = 0
        
        max_c = np.max(np.abs(C))
        
        return max_c / (S * S)

    best_score = compute_score(x)
    best_x = x.copy()
    
    # Local search
    while time.time() - start_time < time_limit:
        # Compute current autocorrelation
        n_fft = 2 * N
        X = np.fft.rfft(x, n=n_fft)
        C = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
        C[0] = 0
        
        max_c = np.max(np.abs(C))
        if max_c == 0:
            break
            
        # Find the lag with the maximum absolute value
        max_lag = np.argmax(np.abs(C))
        target_val = C[max_lag]
        
        # We want to reduce |target_val|.
        # If target_val > 0, we want to flip pairs (i, i+max_lag) where x[i]*x[i+max_lag] == 1.
        # If target_val < 0, we want to flip pairs where x[i]*x[i+max_lag] == -1.
        
        # Candidate indices to flip
        # i ranges from 0 to N - max_lag - 1
        # We flip x[i]. This changes C[max_lag] by -2 * x[i] * x[i+max_lag]
        
        # Collect candidates that move C[max_lag] towards 0
        candidates = []
        for i in range(N - max_lag):
            prod = x[i] * x[i + max_lag]
            # We want the change -2*prod to have opposite sign to target_val
            # If target_val > 0, we need change < 0 => -2*prod < 0 => prod > 0
            # If target_val < 0, we need change > 0 => -2*prod > 0 => prod < 0
            if (target_val > 0 and prod > 0) or (target_val < 0 and prod < 0):
                candidates.append(i)
                
        if not candidates:
            # No direct improvement for this lag, try random flip
            i = np.random.randint(N)
            x[i] *= -1
            new_score = compute_score(x)
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
            else:
                x[i] *= -1 # revert
            continue
            
        # Pick a random candidate
        i = candidates[np.random.randint(len(candidates))]
        
        # Flip
        x[i] *= -1
        
        # Evaluate
        new_score = compute_score(x)
        
        if new_score < best_score:
            best_score = new_score
            best_x = x.copy()
        else:
            # Revert
            x[i] *= -1
            
        # If we've been stuck for a while, maybe add some noise?
        # For now, simple greedy is good.
        
    return best_x
# EVOLVE-BLOCK-END
