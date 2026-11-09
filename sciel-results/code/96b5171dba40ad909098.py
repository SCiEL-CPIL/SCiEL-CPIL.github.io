import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate Golay-Rudin-Shapiro sequence of length 32
    # g(n) = (-1)^{popcount(n)}
    def grs(n):
        # Count set bits
        return 1 if bin(n).count('1') % 2 == 0 else -1
        
    # We need 24 coefficients. Let's generate 32 to have a full GRS block if needed, 
    # but the task asks for 24 coefficients.
    # The optimal polynomial for N=24 might be a truncated GRS or a specific search result.
    # Let's start with the first 24 terms of the GRS sequence.
    p = [grs(i) for i in range(24)]
    
    def evaluate(coeffs):
        # Compute max |P(z)| for |z|=1
        # Use dense sampling
        N = len(coeffs)
        # 10000 points is sufficient for 1e-9 accuracy with refinement
        thetas = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
        # P(z) = sum c_k z^k
        # Use FFT for efficiency? N=24 is small, direct sum is fine.
        # z = exp(i * theta)
        # P(theta) = sum c_k exp(i * k * theta)
        # We can compute this for all thetas at once using matrix multiplication or broadcasting.
        # k = 0..23
        k = np.arange(N)
        # shape (10000, 24)
        exponents = np.exp(1j * np.outer(thetas, k))
        # coeffs shape (24,)
        values = exponents @ np.array(coeffs)
        return np.max(np.abs(values))
        
    best_p = p[:]
    best_score = evaluate(best_p)
    
    # Local search: flip bits and see if score improves
    # Since N=24, there are 24 bits to flip.
    # We can do a few rounds of best-improvement local search.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_score = best_score
        for i in range(24):
            # Flip bit i
            trial_p = best_p[:]
            trial_p[i] = -trial_p[i]
            trial_score = evaluate(trial_p)
            if trial_score < current_score:
                best_p = trial_p
                best_score = trial_score
                current_score = trial_score
                improved = True
    
    # If we have time, do a random restart or more thorough search
    # But GRS is very strong. 
    # Let's verify if the score matches the target 1.2583758 * sqrt(24)
    target_max = 1.2583758 * np.sqrt(24)
    
    # If we haven't reached the target, try a few random restarts
    if best_score > target_max * 1.0001:
        import random
        for _ in range(50):
            if time.time() > deadline: break
            random_p = [1 if random.random() > 0.5 else -1 for _ in range(24)]
            # Local search from random start
            cur_p = random_p[:]
            cur_score = evaluate(cur_p)
            imp = True
            while imp and time.time() < deadline:
                imp = False
                for i in range(24):
                    trial_p = cur_p[:]
                    trial_p[i] = -trial_p[i]
                    trial_score = evaluate(trial_p)
                    if trial_score < cur_score:
                        cur_p = trial_p
                        cur_score = trial_score
                        imp = True
            if cur_score < best_score:
                best_p = cur_p
                best_score = cur_score
                
    return best_p
