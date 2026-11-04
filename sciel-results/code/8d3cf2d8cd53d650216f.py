import numpy as np

N = 100
M = N // 2

def construct():
    """
    Construct the sequence using a discrete local search (hill climbing) on binary
    configurations. This treats the problem as optimizing the positions of domain
    walls (boundaries between 0s and 1s) to minimize the maximum overlap.
    """
    
    def calc_max_overlap(seq):
        """
        Calculate the maximum circular cross-correlation C_k = sum seq[i]*(1-seq[i-k])
        using FFT for efficiency.
        """
        fA = np.fft.fft(seq)
        fB = np.fft.fft(1.0 - seq)
        corr = np.fft.ifft(fA * np.conj(fB)).real
        # corr[k] is the correlation for shift k
        # We want the max for k=1..N-1
        return np.max(corr[1:])

    # Initialize with an alternating pattern (known good starting point)
    x = np.zeros(N)
    x[1::2] = 1.0
    
    best_x = x.copy()
    best_score = calc_max_overlap(best_x)
    
    # Local Search Parameters
    max_iterations = 5000
    np.random.seed(42)
    
    # Precompute indices for faster swapping?
    # We'll just use random swaps and check validity.
    
    for _ in range(max_iterations):
        # Find a random 0 and a random 1 to swap
        zeros = np.where(x == 0)[0]
        ones = np.where(x == 1)[0]
        
        if len(zeros) == 0 or len(ones) == 0:
            break
            
        i0 = zeros[np.random.randint(len(zeros))]
        i1 = ones[np.random.randint(len(ones))]
        
        # Perform swap
        x[i0] = 1.0
        x[i1] = 0.0
        
        score = calc_max_overlap(x)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        else:
            # Revert
            x[i0] = 0.0
            x[i1] = 1.0
            
    # Ensure exact sum and binary values (should already be)
    # The search maintains sum N/2 and binary values.
    return best_x
