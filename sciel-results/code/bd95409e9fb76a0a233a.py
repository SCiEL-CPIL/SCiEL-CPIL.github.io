"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Start with all ones (known good baseline)
    x = np.ones(N)
    
    # Compute current score
    def compute_score(x):
        autoconv = np.convolve(x, x, mode='full')
        s = np.sum(x)
        if abs(s) < 1e-10:
            return np.inf
        return np.max(np.abs(autoconv)) / (s * s)
    
    best_x = x.copy()
    best_score = compute_score(best_x)
    
    # Gradient-based refinement: try flipping individual bits
    # Use a smart order: flip bits that most reduce peak autoconv
    # Precompute contribution of each position to autoconv
    
    # For linear convolution: autoconv[k] = sum_i x[i]*x[i+k-N+1] ... 
    # Actually for np.convolve(x, x, 'full'), result[k] = sum_i x[i]*x[k-i] for valid range
    
    # Let's use a different approach: iterative local search with momentum
    # Try flipping each bit and keep improvements
    
    # First, let's try a spectral approach: 
    # We want to minimize max|F^{-1}(|F(x)|^2)| / |sum(x)|^2
    # where F is DFT. For linear conv, it's more complex.
    
    # Let's do iterative refinement:
    # 1. Compute autoconv
    # 2. Find the lag with maximum |autoconv|
    # 3. Try to reduce it by flipping bits
    
    max_iter = 500
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        autoconv = np.convolve(x, x, mode='full')
        abs_autoconv = np.abs(autoconv)
        max_idx = np.argmax(abs_autoconv)
        max_val = abs_autoconv[max_idx]
        
        # The autoconv at lag max_idx (0-indexed in full conv) 
        # corresponds to shift = max_idx - (N-1)
        shift = max_idx - (N - 1)
        
        # autoconv[max_idx] = sum_i x[i] * x[i + shift] (with appropriate bounds)
        # Flipping x[j] changes this by -2*x[j]*x[j+shift] (if both in range)
        
        # Try flipping each bit and see which reduces the peak most
        # This is O(N^2) per iteration, too slow for N=400, 500 iters
        
        # Instead: greedy approach - flip the bit that most reduces the current peak
        # while not increasing other peaks too much
        
        # Simpler: random restarts with local search
        # Let's try: flip random bits, keep if score improves
        
        # Compute current score
        current_score = compute_score(x)
        
        # Try flipping a random subset of bits
        n_flips = np.random.randint(1, 10)
        indices = np.random.choice(N, n_flips, replace=False)
        x_trial = x.copy()
        x_trial[indices] *= -1
        
        trial_score = compute_score(x_trial)
        
        if trial_score < current_score:
            x = x_trial
            if trial_score < best_score:
                best_score = trial_score
                best_x = x.copy()
        else:
            # Simulated annealing style: accept worse with small probability early on
            if iteration < 100 and np.random.random() < 0.1:
                x = x_trial
    
    # Final polish: single-bit flips
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline:
                break
            i = np.random.randint(N)
            x_trial = x.copy()
            x_trial[i] *= -1
            trial_score = compute_score(x_trial)
            current_score = compute_score(x)
            if trial_score < current_score:
                x = x_trial
                if trial_score < best_score:
                    best_score = trial_score
                    best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
