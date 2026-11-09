"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s limit
    
    best_count = 0
    best_a = [1, 7, 13]
    best_b = [4, 10, 16]
    
    # Precompute GCD table for efficiency
    max_val = 256  # max sum is 128+128
    gcd_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i, max_val + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd(i, j)
    
    def is_pairwise_coprime(nums):
        """Check if all distinct values in nums are pairwise coprime."""
        distinct = list(set(nums))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[distinct[i]][distinct[j]] != 1:
                    return False
        return True
    
    # Search over all combinations of 3 elements from [1, 128]
    # C(128,3) = 349,280 for each set, so total pairs is ~1.2e11 which is too many
    # Need a smarter approach
    
    # Strategy: iterate over A, and for each A, iterate over B
    # But that's still too many. Let's use a different approach.
    
    # Better: for each A (sorted triple), compute the structure of sums
    # The sums are: a[0]+b[0], a[0]+b[1], a[0]+b[2], a[1]+b[0], a[1]+b[1], a[1]+b[2], a[2]+b[0], a[2]+b[1], a[2]+b[2]
    
    # Let's try a heuristic: use small numbers and look for coprime patterns
    
    # Alternative: randomly sample many pairs and check
    import random
    random.seed(42)
    
    # First, try some structured approaches
    # Approach 1: Use sets where elements are spaced to create coprime sums
    
    # Let's try a systematic but limited search
    # Iterate over A from small values, and for each A, sample B values
    
    range_val = 128
    all_values = list(range(1, range_val + 1))
    
    # Try all A from a smaller range first, then expand
    # For efficiency, let's sample B for each A
    
    iterations = 0
    max_iterations = 50000000  # cap on total work
    
    for a0 in range(1, min(50, range_val)):  # start with small a0
        for a1 in range(a0 + 1, min(80, range_val)):
            for a2 in range(a1 + 1, min(128, range_val)):
                A = [a0, a1, a2]
                # For this A, sample various B
                # The sums are a[i] + b[j]
                # We want these to be pairwise coprime
                
                # Try B values that are relatively small or have specific properties
                # Sample B from a subset
                b_candidates = list(range(1, min(60, range_val + 1)))
                
                for b0 in b_candidates:
                    for b1 in range(b0 + 1, min(80, range_val + 1)):
                        for b2 in range(b1 + 1, min(128, range_val + 1)):
                            B = [b0, b1, b2]
                            sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                            distinct_sums = set(sums)
                            if len(distinct_sums) <= best_count:
                                continue
                            if is_pairwise_coprime(sums):
                                best_count = len(distinct_sums)
                                best_a = A[:]
                                best_b = B[:]
                                if best_count == 9:
                                    return [best_a, best_b]
                            
                            iterations += 1
                            if iterations > max_iterations:
                                return [best_a, best_b]
                
                if time.time() > deadline:
                    return [best_a, best_b]
        
        if time.time() > deadline:
            return [best_a, best_b]
    
    # If we haven't found 9, try random sampling
    for _ in range(1000000):
        if time.time() > deadline:
            break
        A = sorted(random.sample(all_values, 3))
        B = sorted(random.sample(all_values, 3))
        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
        distinct_sums = set(sums)
        if len(distinct_sums) <= best_count:
            continue
        if is_pairwise_coprime(sums):
            best_count = len(distinct_sums)
            best_a = A[:]
            best_b = B[:]
            if best_count == 9:
                return [best_a, best_b]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
