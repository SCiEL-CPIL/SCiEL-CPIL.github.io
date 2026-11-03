"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import math
import random
import time


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    Uses chaotic logistic map sequences as seeds for local search optimization
    of the third-order autocorrelation bound (C3).
    """
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3, defined as the maximum absolute value
        of the third-order autocorrelation coefficients.
        """
        # Pad to power of 2 for FFT efficiency
        # Convolution of 3 sequences of length N has length 3N-2.
        N_pad = 1 << (3 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))
    
    def generate_chaotic_signs(seed_val):
        """Generate sign sequence using logistic map at r=4."""
        x = seed_val
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
        return signs
    
    # Generate multiple chaotic candidates and select the best one
    seeds = [
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        np.pi % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.e % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1/np.sqrt(2)) % 1.0,
        np.sqrt(5) % 1.0,
        np.sqrt(7) % 1.0
    ]
    
    best_signs = None
    best_bound = float('inf')
    
    for seed in seeds:
        signs = generate_chaotic_signs(seed)
        bound = compute_c3_bound(signs)
        
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()
    
    # Local search with simulated annealing
    signs = best_signs
    current_bound = best_bound
    
    # Time budget for search
    start_time = time.time()
    max_time = 110  # Leave 10s buffer
    
    # Simulated Annealing parameters
    T_start = 2.0
    T_end = 0.01
    alpha = 0.9995
    T = T_start
    evals = 0
    max_evals = 100000
    
    # Set seed for reproducibility
    random.seed(12345)
    
    # Precompute FFT-related quantities for faster evaluation
    N_pad = 1 << (3 * N - 1).bit_length()
    
    while T > T_end and evals < max_evals and (time.time() - start_time) < max_time:
        # Pick a random index to flip
        i = random.randint(0, N - 1)
        
        # Flip
        signs[i] *= -1.0
        new_bound = compute_c3_bound(signs)
        evals += 1
        
        delta = new_bound - current_bound
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_bound = new_bound
        else:
            # Revert
            signs[i] *= -1.0
            
        T *= alpha
        
    # Final greedy pass to polish
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        for i in range(N):
            if (time.time() - start_time) >= max_time:
                break
            signs[i] *= -1.0
            new_bound = compute_c3_bound(signs)
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                signs[i] *= -1.0
                
    return signs
# EVOLVE-BLOCK-END
