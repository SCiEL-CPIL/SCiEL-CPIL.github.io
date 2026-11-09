"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Deadline: 110 seconds (safety margin for 120s limit)
    start_time = time.time()
    time_limit = 110.0
    
    rng = np.random.default_rng(42)
    
    # Structural Change: Initialize with Thue-Morse sequence
    # Thue-Morse has low autocorrelation sidelobes compared to random.
    # t_n = popcount(n) % 2
    # Map 0 -> 1, 1 -> -1
    a = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    
    # Define score function: Sum of squared sidelobes of circular autocorrelation
    def compute_score(a):
        fa = np.fft.fft(a)
        power = fa * np.conj(fa)
        R = np.fft.ifft(power).real
        # Exclude lag 0 (R[0] is N^2)
        # R is circular, so R[0] is the peak.
        # Sidelobes are R[1:] and R[-(N-1):]? 
        # Actually, for real a, R is symmetric. R[k] = R[N-k].
        # The sum of squares of all non-zero lags is sum(R[1:]**2) + sum(R[:N-1]... wait.
        # R[0] is lag 0.
        # R[1] to R[N-1] are lags 1 to N-1.
        # So sum(R[1:]**2) covers lags 1..N-1.
        return np.sum(R[1:]**2)

    current_a = a.copy()
    current_score = compute_score(current_a)
    best_a = current_a.copy()
    best_score = current_score
    
    # Simulated Annealing
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    num_iterations = 500000  # Will be limited by time
    
    # Precompute indices for random flips
    # We flip single bits to allow fine-grained search
    
    for i in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature decay (linear or exponential)
        # Exponential is often better
        T = T_start * (T_end / T_start) ** (i / num_iterations)
        
        # Generate candidate: flip one random bit
        idx = rng.integers(0, N)
        candidate_a = current_a.copy()
        candidate_a[idx] *= -1.0
        
        candidate_score = compute_score(candidate_a)
        delta = candidate_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_a = candidate_a
            current_score = candidate_score
            
            if current_score < best_score:
                best_score = current_score
                best_a = current_a.copy()
                
    return best_a
# EVOLVE-BLOCK-END
