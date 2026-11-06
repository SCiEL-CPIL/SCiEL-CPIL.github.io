"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    max_sum = 256
    
    # Precompute GCD table
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_pairwise_coprime(nums):
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[nums[i]][nums[j]] != 1:
                    return False
        return True
    
    # Key optimization: early termination in sum construction
    def check_partial(A, B_partial, sums_so_far):
        """Check if adding more to B will still allow pairwise coprime sums."""
        for s in sums_so_far:
            if s < 1 or s > max_sum:
                return False
        # Check pairwise coprime among sums_so_far
        n = len(sums_so_far)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[sums_so_far[i]][sums_so_far[j]] != 1:
                    return False
        return True
    
    # Strategy: iterate over A, then build B element by element with pruning
    # For each A, try all B1, then for each B1 try all B2, then for each (B1,B2) try all B3
    
    # Limit search space
    search_range = list(range(1, 80))
    
    # Precompute for a given A, which B values are "compatible"
    # i.e., a_i + b is coprime to all other a_j + b' for j != i and valid b'
    
    # Actually, let's just do a well-optimized triple loop with early pruning
    
    best_result = None
    best_distinct_count = 0
    
    for a0 in range(1, 50):
        if time.time() > deadline:
            break
        for a1 in range(a0 + 1, 50):
            if time.time() > deadline:
                break
            for a2 in range(a1 + 1, 50):
                if time.time() > deadline:
                    break
                A = (a0, a1, a2)
                
                # For each candidate B, check validity
                # B must have 3 distinct elements
                for b0 in range(1, 50):
                    if time.time() > deadline:
                        break
                    s00 = a0 + b0
                    s10 = a1 + b0
                    s20 = a2 + b0
                    if s00 > max_sum or s10 > max_sum or s20 > max_sum:
                        continue
                    if gcd_table[s00][s10] != 1 or gcd_table[s00][s20] != 1 or gcd_table[s10][s20] != 1:
                        continue
                    
                    for b1 in range(b0 + 1, 50):
                        if time.time() > deadline:
                            break
                        s01 = a0 + b1
                        s11 = a1 + b1
                        s21 = a2 + b1
                        if s01 > max_sum or s11 > max_sum or s21 > max_sum:
                            continue
                        # Check new sums against existing
                        existing = [s00, s10, s20]
                        new_sums = [s01, s11, s21]
                        valid = True
                        for ns in new_sums:
                            for es in existing:
                                if gcd_table[ns][es] != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        # Check among new_sums
                        if gcd_table[s01][s11] != 1 or gcd_table[s01][s21] != 1 or gcd_table[s11][s21] != 1:
                            continue
                        
                        for b2 in range(b1 + 1, 50):
                            if time.time() > deadline:
                                break
                            s02 = a0 + b2
                            s12 = a1 + b2
                            s22 = a2 + b2
                            if s02 > max_sum or s12 > max_sum or s22 > max_sum:
                                continue
                            existing2 = [s00, s10, s20, s01, s11, s21]
                            new_sums2 = [s02, s12, s22]
                            valid2 = True
                            for ns in new_sums2:
                                for es in existing2:
                                    if gcd_table[ns][es] != 1:
                                        valid2 = False
                                        break
                                if not valid2:
                                    break
                            if not valid2:
                                continue
                            if gcd_table[s02][s12] != 1 or gcd_table[s02][s22] != 1 or gcd_table[s12][s22] != 1:
                                continue
                            
                            # All 9 sums are pairwise coprime!
                            # Check they're all distinct
                            all_sums = [s00, s01, s02, s10, s11, s12, s20, s21, s22]
                            if len(set(all_sums)) == 9:
                                return [[a0, a1, a2], [b0, b1, b2]]
    
    # Fallback
    return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
