"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    Minimizes the maximum absolute value of the third-order autocorrelation (C3).
    """
    
    # Pre-calculate FFT parameters
    # Convolution of 3 sequences of length N has length 3N-2
    conv_len = 3 * N - 2
    N_pad = 1 << (conv_len - 1).bit_length()
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3 (max abs of triple autocorrelation).
        """
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    # 1. Initialization
    # Use a fixed seed for reproducibility, but try a few different initializations
    best_signs = None
    best_bound = float('inf')
    
    # Try a few different initial sequences
    # 1. Random
    # 2. Thue-Morse (known for low discrepancy)
    # 3. Logistic Map (from previous attempt, but maybe better seeds)
    
    initial_candidates = []
    
    # Random
    rng = np.random.default_rng(42)
    initial_candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Thue-Morse
    def thue_morse(n):
        t = 0
        seq = []
        for i in range(n):
            # Count trailing zeros in binary representation of i
            # Or use the parity of the number of 1s in binary
            seq.append(1.0 if bin(i).count('1') % 2 == 0 else -1.0)
        return np.array(seq)
    initial_candidates.append(thue_morse(N))
    
    # Logistic Map with a good seed
    x = 0.123456789
    signs = np.zeros(N)
    for i in range(N):
        x = 4.0 * x * (1.0 - x)
        signs[i] = 1.0 if x > 0.5 else -1.0
    initial_candidates.append(signs)
    
    # 2. Optimization Phase
    # We will run a local search on each candidate and keep the best
    
    start_time = time.time()
    timeout = 115.0  # Leave 5 seconds buffer
    
    for init_signs in initial_candidates:
        signs = init_signs.copy()
        current_bound = compute_c3_bound(signs)
        
        # Greedy Hill Climbing
        improved = True
        while improved:
            improved = False
            for i in range(N):
                if time.time() - start_time > timeout:
                    break
                signs[i] *= -1.0
                new_bound = compute_c3_bound(signs)
                if new_bound < current_bound:
                    current_bound = new_bound
                    improved = True
                else:
                    signs[i] *= -1.0
            if time.time() - start_time > timeout:
                break
        
        # Simulated Annealing
        if time.time() - start_time < timeout - 5.0:  # Ensure some time for SA
            T_start = 1.0
            T_end = 0.01
            alpha = 0.995
            max_evals_sa = 20000
            
            # Adjust alpha to fit within time if possible, or just use max_evals
            # Estimate time per eval
            # We'll just run until time or max_evals
            
            T = T_start
            evals = 0
            random.seed(123)
            
            while T > T_end and evals < max_evals_sa:
                if time.time() - start_time > timeout:
                    break
                
                i = random.randint(0, N - 1)
                signs[i] *= -1.0
                new_bound = compute_c3_bound(signs)
                evals += 1
                
                delta = new_bound - current_bound
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_bound = new_bound
                else:
                    signs[i] *= -1.0
                    
                T *= alpha
            
            # Final Greedy Pass
            improved = True
            while improved:
                improved = False
                for i in range(N):
                    if time.time() - start_time > timeout:
                        break
                    signs[i] *= -1.0
                    new_bound = compute_c3_bound(signs)
                    if new_bound < current_bound:
                        current_bound = new_bound
                        improved = True
                    else:
                        signs[i] *= -1.0
                if time.time() - start_time > timeout:
                    break
        
        if current_bound < best_bound:
            best_bound = current_bound
            best_signs = signs.copy()
            
        if time.time() - start_time > timeout:
            break
            
    return best_signs
# EVOLVE-BLOCK-END
