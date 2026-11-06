"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # The sum is sum_{i=1}^{30} c[i-1]/i where c[i] in {-1, 0, 1}
    # We want to minimize |sum| subject to sum != 0
    
    # Strategy: Use local search / hill climbing with random restarts
    # The key insight is that we have 30 variables each in {-1,0,1}
    # Total space is 3^30 which is too large for brute force
    # But we can do smart local search
    
    # Precompute the weights
    weights = [1.0 / i for i in range(1, 31)]
    
    # Use Fraction for exact arithmetic to avoid floating point issues
    # But for speed during search, use float and verify with Fraction at the end
    
    best_abs = float('inf')
    best_coeffs = None
    
    def compute_sum_float(coeffs):
        s = 0.0
        for c, w in zip(coeffs, weights):
            if c != 0:
                s += c * w
        return s
    
    def compute_sum_frac(coeffs):
        s = Fraction(0)
        for c, w_num in zip(coeffs, range(1, 31)):
            if c != 0:
                s += Fraction(c, w_num)
        return s
    
    def evaluate(coeffs):
        """Return absolute value of sum, or 0 if sum is exactly zero"""
        s = compute_sum_frac(coeffs)
        if s == 0:
            return 0  # invalid
        return abs(s)
    
    def local_search(coeffs):
        """Hill climbing: try flipping each coefficient to the other two values"""
        current = list(coeffs)
        current_sum = compute_sum_float(current)
        current_abs = abs(current_sum)
        
        improved = True
        while improved:
            improved = False
            for i in range(30):
                for new_val in [-1, 0, 1]:
                    if new_val == current[i]:
                        continue
                    old_val = current[i]
                    current[i] = new_val
                    new_sum = current_sum + (new_val - old_val) * weights[i]
                    new_abs = abs(new_sum)
                    if new_abs < current_abs:
                        # Check if it's exactly zero using Fraction
                        s = compute_sum_frac(current)
                        if s == 0:
                            current[i] = old_val
                            continue
                        current_sum = new_sum
                        current_abs = new_abs
                        improved = True
                    else:
                        current[i] = old_val
        return current, current_abs
    
    # Multiple random restarts
    random.seed(42)
    
    # Start with some good initial guesses
    # 1) All zeros except one +1 and one -1 (like the seed)
    # 2) Random assignments
    # 3) Alternating patterns
    
    initial_guesses = []
    
    # Seed-like: [0]*28 + [1,-1]
    initial_guesses.append([0]*28 + [1, -1])
    
    # Alternating: 1, -1, 1, -1, ...
    initial_guesses.append([1 if i % 2 == 0 else -1 for i in range(30)])
    
    # Random restarts
    for _ in range(200):
        if time.time() > deadline:
            break
        initial_guesses.append([random.choice([-1, 0, 1]) for _ in range(30)])
    
    # Also try some structured patterns
    # Pairs that cancel: 1/1 - 1/2 + 1/3 - 1/4 ...
    for n in range(1, 30):
        pattern = [0]*30
        for i in range(n):
            pattern[i] = 1 if i % 2 == 0 else -1
        initial_guesses.append(pattern)
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        # Ensure the sum is not zero
        s = compute_sum_frac(guess)
        if s == 0:
            # Perturb it
            guess = list(guess)
            if guess[0] != 1:
                guess[0] = 1
            elif guess[0] == 1:
                guess[0] = -1
            s = compute_sum_frac(guess)
            if s == 0:
                continue
        
        result, result_abs = local_search(guess)
        
        # Verify with exact arithmetic
        s = compute_sum_frac(result)
        if s == 0:
            continue
        exact_abs = abs(s)
        
        if exact_abs < best_abs or best_coeffs is None:
            best_abs = exact_abs
            best_coeffs = result
    
    # If we still haven't found anything, use the seed
    if best_coeffs is None:
        best_coeffs = [0]*28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
