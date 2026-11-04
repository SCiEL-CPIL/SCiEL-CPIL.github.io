import numpy as np

N = 400

def construct():
    # Target size for the support.
    # Sparse design as per directive.
    # Sidon sets are optimal for off-peak, but let's try random + local search.
    # If Sidon scores 0.405, maybe a better sparse set exists or the metric is different.
    # We will minimize the maximum off-peak autocorrelation.
    
    M = 20  # Keep similar sparsity to the seed to be competitive
    
    # Random initialization
    support = np.random.choice(N, M, replace=False)
    signs = np.random.choice([-1, 1], M)
    
    # Function to compute max off-peak autocorrelation
    def get_peak(support, signs):
        x = np.zeros(N)
        x[support] = signs
        # Normalize to L2=1 for consistent comparison
        x = x / np.sqrt(M)
        
        # Compute autocorrelation
        # Circular autocorrelation
        # R[d] = sum x[k] x[k+d]
        # Using FFT
        X = np.fft.fft(x)
        R = np.fft.ifft(np.abs(X)**2).real
        # R[0] is 1. We want max of R[1:]
        # Note: R is symmetric.
        peak = np.max(np.abs(R[1:]))
        return peak

    current_peak = get_peak(support, signs)
    
    # Local Search
    # Try swapping indices or flipping signs
    for _ in range(1000): # Iterations
        improved = False
        
        # Try flipping a sign
        idx = np.random.randint(M)
        old_sign = signs[idx]
        signs[idx] = -old_sign
        new_peak = get_peak(support, signs)
        if new_peak < current_peak:
            current_peak = new_peak
            improved = True
        else:
            signs[idx] = old_sign
            
        # Try swapping a support index with a random non-support index
        idx_in = np.random.randint(M)
        idx_out = np.random.choice(N, replace=False)
        if idx_out not in support:
            old_val = support[idx_in]
            support[idx_in] = idx_out
            new_peak = get_peak(support, signs)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                support[idx_in] = old_val
        
        if not improved and _ > 500:
            # Restart with new random initialization if stuck
            support = np.random.choice(N, M, replace=False)
            signs = np.random.choice([-1, 1], M)
            current_peak = get_peak(support, signs)
            
    # Construct final x
    x = np.zeros(N)
    x[support] = signs / np.sqrt(M)
    
    return x
