"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    max_val = 256
    
    def all_pairwise_coprime(values):
        """Check if all distinct values are pairwise coprime."""
        vals = list(set(values))
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    def evaluate(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if all_pairwise_coprime(distinct):
            return len(distinct)
        return 0

    # Precompute primes up to 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(max_val)
    
    # Systematic search for 9 distinct primes
    # Iterate over A
    range_128 = range(1, 129)
    
    # To speed up, we can restrict A to smaller values first, 
    # but let's try a smarter check.
    
    # We will iterate over all A.
    # For each A, we will try to find B.
    # To make it faster, we check candidates for b1, b2, b3.
    
    # Optimization: If we find a score of 9, we can break.
    
    # Let's iterate over A combinations
    # C(128, 3) is ~345k.
    # For each A, we don't want to iterate all B (another 345k).
    # Instead, we pick b1, b2 from [1, 128].
    # Check if a_i + b1 and a_i + b2 are primes for all i.
    # If so, collect these 6 sums. Check if they are distinct and pairwise coprime.
    # Then iterate b3 from [1, 128].
    # Check if a_i + b3 are primes for all i.
    # Check if the 9 sums are distinct and pairwise coprime.
    
    # This is 345k * 128 * 128 * 128 which is too much.
    # 345k * 2M = 700 billion. Too slow.
    
    # We need a better heuristic.
    # Random search with seed 42 found a solution (score 1.8 implies 9 sums).
    # So a solution exists.
    # Let's just do a fast random search to find it quickly, then verify.
    # Or, restrict A to a smaller range, e.g., [1, 30].
    
    import random
    random.seed(42)
    
    # Try random search first, it's fast and likely to find 9.
    start_time = time.time()
    attempts = 0
    while time.time() < deadline:
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        
        score = evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
            if best_score == 9:
                break
        attempts += 1
        
        # If we haven't found 9 after some attempts, try a more structured approach?
        # The random search with seed 42 previously found 9 (score 1.8).
        # So it should find it quickly.
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
