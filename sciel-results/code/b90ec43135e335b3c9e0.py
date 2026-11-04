"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_peak(x):
    """Compute max absolute value of autoconvolution of x."""
    # For step function on N intervals, autoconvolution has 2N-1 bins
    # Use FFT for efficiency
    # Pad to power of 2 for efficient FFT
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    f = np.fft.rfft(x, n_fft)
    conv = np.fft.irfft(f * np.conj(f), n_fft)
    # The autoconvolution is centered; take max abs
    return np.max(np.abs(conv))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Start with all ones (seed)
    x = np.ones(N)
    best_x = x.copy()
    best_score = _autoconv_peak(best_x)
    
    # Sparse combinatorial design: select subset of intervals to be non-zero
    # with optimized signs via greedy local search
    
    # Try different sparsity levels
    sparsity_levels = [50, 100, 150, 200, 250, 300, 350]
    
    for k in sparsity_levels:
        if time.time() > deadline:
            break
            
        # Random initialization of support set
        support = np.random.choice(N, size=k, replace=False)
        support.sort()
        
        # Initialize with random signs
        signs = np.random.choice([-1, 1], size=N)
        x_trial = signs.copy()
        x_trial[np.array([i for i in range(N) if i not in support])] = 0.0
        
        # Local search: flip individual signs
        for iteration in range(5000):
            if time.time() > deadline:
                break
                
            # Try flipping a random active interval
            idx = np.random.choice(support)
            old_sign = x_trial[idx]
            x_trial[idx] = -old_sign
            
            new_score = _autoconv_peak(x_trial)
            
            if new_score < best_score:
                best_score = new_score
                best_x = x_trial.copy()
            else:
                x_trial[idx] = old_sign  # revert
        
        # Also try adding/removing intervals from support
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            if np.random.random() < 0.5 and k < N:
                # Add an interval
                inactive = np.array([i for i in range(N) if i not in support])
                if len(inactive) == 0:
                    continue
                idx = np.random.choice(inactive)
                support = np.sort(np.append(support, idx))
                k += 1
                x_trial[idx] = np.random.choice([-1, 1])
            elif k > 10:
                # Remove an interval
                idx = np.random.choice(support)
                support = np.delete(support, np.where(support == idx)[0][0])
                k -= 1
                x_trial[idx] = 0.0
            
            new_score = _autoconv_peak(x_trial)
            if new_score < best_score:
                best_score = new_score
                best_x = x_trial.copy()
    
    # Final polish: local search on best solution
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        idx = np.random.choice(N)
        old_val = best_x[idx]
        new_val = -old_val if old_val != 0 else np.random.choice([-1, 1])
        best_x[idx] = new_val
        
        new_score = _autoconv_peak(best_x)
        if new_score >= best_score:
            best_x[idx] = old_val  # revert
    
    return best_x
# EVOLVE-BLOCK-END
