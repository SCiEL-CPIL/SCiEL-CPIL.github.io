"""
Strategy: Simulated Annealing for Signed Step Function Autoconvolution Minimization.
Different from spectral phase retrieval which gets stuck in local minima due to hard sign constraints.
SA explores the space of sign flips, accepting worse states to escape local optima.
"""

import numpy as np
import time

N = 400

def _compute_score(x):
    """
    Compute the metric: max|autoconvolution| / (integral)^2.
    For a binary sequence x (values +/-1), the integral is sum(x) or ||x||_2?
    The prompt says "squared integral". 
    Seed program uses np.ones(N). 
    If x = ones(N):
      autoconv[0] = N. max|autoconv| = N.
      If integral = sum(x) = N, score = N / N^2 = 1/N.
      If integral = ||x||_2 = sqrt(N), score = N / (sqrt(N))^2 = N/N = 1.
      
    The seed score is 1.0. 
    If the metric was 1/N, the seed score would be 1/400 = 0.0025.
    Since the seed scores 1.0, the metric must be such that ones(N) yields 1.0.
    max|autoconv| / ||x||_2^2 = N / N = 1.0.
    So the metric is max|autoconv| / sum(x^2).
    """
    # Use FFT for fast circular autocorrelation
    X = np.fft.rfft(x)
    power = X * np.conj(X)
    acf = np.fft.irfft(power, N)
    
    max_abs_acf = np.max(np.abs(acf))
    
    # Normalizer: squared L2 norm (sum of squares)
    # For binary x, sum(x^2) = N.
    # If we allow non-binary, this scales. But "signed step heights" usually implies +/-1.
    # The prompt says "sequence of signed step heights". 
    # Let's assume the output should be +/-1 to match the "signed step" description and the seed.
    norm_sq = np.sum(x * x)
    
    if norm_sq < 1e-12:
        return np.inf
        
    return max_abs_acf / norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with random signs
    rng = np.random.RandomState(42)
    x = rng.choice([-1, 1], size=N)
    
    # Initial score
    current_score = _compute_score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    # We want to minimize the score.
    # Start with high temperature to allow large changes, cool down.
    T_start = 0.5  # Arbitrary scale, will adjust based on score differences
    T_end = 1e-6
    # Number of iterations depends on time. 
    # Each iteration is an FFT (O(N log N)) and a score calc.
    # N=400 is small. Let's estimate 10000 iterations in 10s? 
    # Actually, let's just loop until time runs out, with adaptive cooling.
    
    # Use a geometric cooling schedule
    # We don't know the total number of steps ahead, so we use a time-based cooling.
    # Or simpler: fixed number of steps per "phase" and reduce T.
    
    # Let's do a fixed number of steps, say 5000, and if time remains, restart with lower T.
    # Or just one long run with exponential decay.
    
    total_steps = 20000  # Will be limited by time
    alpha = 0.999  # Cooling factor per step
    
    T = T_start
    
    for i in range(total_steps):
        if time.time() > deadline:
            break
            
        # Pick a random index to flip
        idx = rng.randint(N)
        
        # Flip sign
        x[idx] *= -1
        
        # Calculate new score
        new_score = _compute_score(x)
        
        # Metropolis criterion
        delta = new_score - current_score
        if delta < 0:
            # Improvement
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            # Worse, accept with probability exp(-delta / T)
            # If T is very small, this is ~0
            if T > 1e-12:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_score = new_score
            else:
                # Revert if not accepted (T too low to accept)
                x[idx] *= -1
                # current_score remains unchanged
        
        # Cool down
        T *= alpha
        
        # If T is too low, we might get stuck. 
        # For a robust search, we could restart or reheat, but simple SA is a good differentiator.
        
    return best_x
