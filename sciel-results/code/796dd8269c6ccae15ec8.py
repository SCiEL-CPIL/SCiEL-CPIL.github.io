"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s kill
    
    best_score = 5
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    def is_pairwise_coprime(vals):
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if math.gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    # Use odd for A, even for B to avoid all-even sums
    odd_vals = list(range(1, 129, 2))
    even_vals = list(range(2, 129, 2))
    
    # Expand search ranges
    A_search = odd_vals[:50]   # 1 to 99
    B_search = even_vals[:50]  # 2 to 100
    
    # Precompute gcd table for faster checks
    max_sum = 128 + 128
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(1, max_sum + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    def is_pairwise_coprime_fast(vals, gt):
        n = len(vals)
        for i in range(n):
            vi = vals[i]
            for j in range(i + 1, n):
                if gt[vi][vals[j]] != 1:
                    return False
        return True
    
    for A in itertools.combinations(A_search, 3):
        if time.time() > deadline:
            return best_result
        a1, a2, a3 = A
        d12 = a2 - a1
        d13 = a3 - a1
        d23 = a3 - a2
        
        valid_b = []
        for b in B_search:
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if (s1 > max_sum or s2 > max_sum or s3 > max_sum):
                continue
            # Check that sums within this B are pairwise coprime
            if gcd_table[s1][s2] == 1 and gcd_table[s1][s3] == 1 and gcd_table[s2][s3] == 1:
                valid_b.append((b, s1, s2, s3))
        
        if len(valid_b) < 3:
            continue
        
        # Limit to top candidates for performance
        if len(valid_b) > 30:
            valid_b = valid_b[:30]
        
        n = len(valid_b)
        for i in range(n):
            if time.time() > deadline:
                return best_result
            b1, s1_1, s2_1, s3_1 = valid_b[i]
            sums1 = [s1_1, s2_1, s3_1]
            for j in range(i + 1, n):
                b2, s1_2, s2_2, s3_2 = valid_b[j]
                sums12 = sums1 + [s1_2, s2_2, s3_2]
                if not is_pairwise_coprime_fast(sums12, gcd_table):
                    continue
                for k in range(j + 1, n):
                    b3, s1_3, s2_3, s3_3 = valid_b[k]
                    all_sums = sums12 + [s1_3, s2_3, s3_3]
                    if len(set(all_sums)) <= best_score:
                        continue
                    if is_pairwise_coprime_fast(all_sums, gcd_table):
                        distinct_count = len(set(all_sums))
                        if distinct_count > best_score:
                            best_score = distinct_count
                            best_result = [list(A), [b1, b2, b3]]
                            if best_score == 9:
                                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
