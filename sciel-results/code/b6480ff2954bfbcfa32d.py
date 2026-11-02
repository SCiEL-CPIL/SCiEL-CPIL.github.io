"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    We use the Logistic Map at r=4 (a chaotic system) to generate the sign sequence.
    Different irrational initial conditions produce different ergodic trajectories.
    We evaluate the C3 bound (max absolute value of the triple autocorrelation)
    for multiple trajectories and select the one with the minimum bound.
    A fast local search is then applied to the best candidate to further minimize the bound.
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

    # Generate candidates using Logistic Map x_{n+1} = 4 x_n (1 - x_n)
    # We try several irrational seeds to explore different chaotic trajectories.
    seeds = [
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        np.pi % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.e % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1/np.sqrt(2)) % 1.0
    ]
    
    best_signs = None
    best_bound = float('inf')
    
    for seed in seeds:
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
            
        bound = compute_c3_bound(signs)
        
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()
            
    # Fast local search on the best candidate
    # We use a greedy approach: flip bits if it reduces the bound.
    # Limit evaluations to ensure speed.
    signs = best_signs
    current_bound = best_bound
    max_evals = 10000
    evals = 0
    improved = True
    
    while improved and evals < max_evals:
        improved = False
        for i in range(N):
            if evals >= max_evals:
                break
            signs[i] *= -1.0
            new_bound = compute_c3_bound(signs)
            evals += 1
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                signs[i] *= -1.0
                
    return signs
# EVOLVE-BLOCK-END
