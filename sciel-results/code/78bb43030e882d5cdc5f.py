"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct the sequence of step heights.
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    """
    # We search for a binary sequence x in {-1, 1}^N with sum 0.
    # h = (x + 1) / 2.
    
    # Initialize with blocks to leverage "domain wall" structure
    half = N // 2
    x = np.zeros(N)
    x[:half] = 1.0
    x[half:] = -1.0
    
    def compute_autocorr_fft(x):
        # Circular autocorrelation using FFT
        fx = np.fft.fft(x)
        acf = np.fft.ifft(fx * np.conj(fx)).real
        return acf
    
    A = compute_autocorr_fft(x)
    # Objective: Maximize min(A[1:]) to minimize max overlap
    current_score = np.min(A[1:])
    
    best_x = x.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 0.01
    steps = 100000 # Increased steps due to O(N log N) cost
    cooling_rate = (T_min / T) ** (1.0 / steps)
    
    for s in range(steps):
        # Pick two random indices i, j to swap (must be different values)
        i = np.random.randint(N)
        j = np.random.randint(N)
        
        if x[i] == x[j]:
            continue
            
        # Perform swap
        x[i], x[j] = x[j], x[i]
        
        # Compute new score
        A_new = compute_autocorr_fft(x)
        new_score = np.min(A_new[1:])
        
        # Acceptance criterion (Maximizing)
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
        else:
            # Boltzmann acceptance
            delta = new_score - current_score
            if np.random.rand() < np.exp(delta / T):
                current_score = new_score
            else:
                # Revert
                x[i], x[j] = x[j], x[i]
                
        T *= cooling_rate
        
    # Convert best_x to h
    h = (best_x + 1.0) / 2.0
    
    return h
# EVOLVE-BLOCK-END
