"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a step function h to minimize the maximum overlap with shifted 1-h.
    Uses a local search approach to flatten the autocorrelation of a binary sequence.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with a random binary sequence with sum N/2
    np.random.seed(42)
    h = np.zeros(N)
    indices = np.random.permutation(N)
    h[indices[:N//2]] = 1.0
    
    best_score = float('inf')
    best_h = h.copy()
    
    def get_score(h):
        # Compute circular autocorrelation
        # C(k) = sum h[i] h[i+k]
        # Overlap S(k) = sum h[i] (1-h[i+k]) = sum h[i] - C(k)
        # We want to minimize max S(k)
        # This is equivalent to minimizing max (N/2 - C(k))
        # Which is equivalent to maximizing min C(k)
        
        # FFT based correlation
        h_float = h.astype(np.float64)
        H = np.fft.fft(h_float)
        corr = np.real(np.fft.ifft(H * np.conj(H)))
        # corr[k] is C(k)
        
        # We only care about k=1 to N-1 (shifts)
        # S(k) = N/2 - corr[k]
        # Max S(k) is at min corr[k]
        min_corr = np.min(corr[1:])
        max_overlap = N/2 - min_corr
        
        return max_overlap

    current_score = get_score(h)
    best_score = current_score
    best_h = h.copy()
    
    # Local search: flip pairs of 0s and 1s to reduce max overlap
    # Or single flips if we maintain sum? No, sum must be N/2.
    # So we must flip a 0 to 1 and a 1 to 0.
    
    while time.time() < deadline:
        # Find a random pair to flip
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        
        if i == j:
            continue
            
        # Try flip
        h[i] = 0.0
        h[j] = 1.0
        
        new_score = get_score(h)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
            # Keep the change
            # Optionally do a hill climb from here
        else:
            # Revert
            h[i] = 1.0
            h[j] = 0.0
            
            # If no improvement for a while, maybe restart or change strategy
            # For simplicity, just continue random search
    
    return best_h
# EVOLVE-BLOCK-END
