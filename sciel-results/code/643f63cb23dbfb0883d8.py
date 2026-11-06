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
    
    max_sum = 256
    
    # Precompute gcd table for faster checks
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_pairwise_coprime_fast(vals, gt):
        n = len(vals)
        for i in range(n):
            vi = vals[i]
            for j in range(i + 1, n):
                if gt[vi][vals[j]] != 1:
                    return False
        return True
    
    # Phase 1: Odd A, Even B (avoids even-even sum collisions)
    odd_vals = list(range(1, 129, 2))
    even_vals = list(range(2, 129, 2))
    
    A_search = odd_vals[:60]   # 1 to 119
    B_search = even_vals[:60]  # 2 to 120
    
    for A in itertools.combinations(A_search, 3):
        if time.time() > deadline:
            return best_result
        a1, a2, a3 = A
        
        valid_b = []
        for b in B_search:
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if s3 > max_sum:
                continue
            # Check that sums within this B are pairwise coprime
            if gcd_table[s1][s2] == 1 and gcd_table[s1][s3] == 1 and gcd_table[s2][s3] == 1:
                valid_b.append((b, s1, s2, s3))
        
        if len(valid_b) < 3:
            continue
        
        # Limit to top candidates for performance
        max_valid = 40
        if len(valid_b) > max_valid:
            valid_b = valid_b[:max_valid]
        
        n = len(valid_b)
        for i in range(n):
            if time.time() > deadline:
                return best_result
            b1, s1_1, s2_1, s3_1 = valid_b[i]
            for j in range(i + 1, n):
                b2, s1_2, s2_2, s3_2 = valid_b[j]
                # Quick check: 6 sums must be pairwise coprime
                if (gcd_table[s1_1][s1_2] != 1 or gcd_table[s1_1][s2_2] != 1 or
                    gcd_table[s1_1][s3_2] != 1 or gcd_table[s2_1][s1_2] != 1 or
                    gcd_table[s2_1][s2_2] != 1 or gcd_table[s2_1][s3_2] != 1 or
                    gcd_table[s3_1][s1_2] != 1 or gcd_table[s3_1][s2_2] != 1 or
                    gcd_table[s3_1][s3_2] != 1):
                    continue
                for k in range(j + 1, n):
                    b3, s1_3, s2_3, s3_3 = valid_b[k]
                    # Check remaining 9 cross-pairs
                    if (gcd_table[s1_1][s1_3] != 1 or gcd_table[s1_1][s2_3] != 1 or
                        gcd_table[s1_1][s3_3] != 1 or gcd_table[s2_1][s1_3] != 1 or
                        gcd_table[s2_1][s2_3] != 1 or gcd_table[s2_1][s3_3] != 1 or
                        gcd_table[s3_1][s1_3] != 1 or gcd_table[s3_1][s2_3] != 1 or
                        gcd_table[s3_1][s3_3] != 1 or
                        gcd_table[s1_2][s1_3] != 1 or gcd_table[s1_2][s2_3] != 1 or
                        gcd_table[s1_2][s3_3] != 1 or gcd_table[s2_2][s1_3] != 1 or
                        gcd_table[s2_2][s2_3] != 1 or gcd_table[s2_2][s3_3] != 1 or
                        gcd_table[s3_2][s1_3] != 1 or gcd_table[s3_2][s2_3] != 1 or
                        gcd_table[s3_2][s3_3] != 1):
                        continue
                    all_sums = [s1_1, s2_1, s3_1, s1_2, s2_2, s3_2, s1_3, s2_3, s3_3]
                    distinct_count = len(set(all_sums))
                    if distinct_count > best_score:
                        best_score = distinct_count
                        best_result = [list(A), [b1, b2, b3]]
                        if best_score == 9:
                            return best_result
    
    # Phase 2: Try some mixed-parity combinations
    all_vals = list(range(1, 129))
    # Sample a subset of all values for mixed search
    A_search_mixed = [v for v in all_vals if v % 3 != 0][:40]
    B_search_mixed = [v for v in all_vals if v % 3 != 0 and v not in A_search_mixed[:15]][:40]
    
    for A in itertools.combinations(A_search_mixed, 3):
        if time.time() > deadline:
            return best_result
        a1, a2, a3 = A
        
        valid_b = []
        for b in B_search_mixed:
            if b in A:
                continue
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if s3 > max_sum:
                continue
            if gcd_table[s1][s2] == 1 and gcd_table[s1][s3] == 1 and gcd_table[s2][s3] == 1:
                valid_b.append((b, s1, s2, s3))
        
        if len(valid_b) < 3:
            continue
        
        max_valid = 30
        if len(valid_b) > max_valid:
            valid_b = valid_b[:max_valid]
        
        n = len(valid_b)
        for i in range(n):
            if time.time() > deadline:
                return best_result
            b1, s1_1, s2_1, s3_1 = valid_b[i]
            for j in range(i + 1, n):
                b2, s1_2, s2_2, s3_2 = valid_b[j]
                if (gcd_table[s1_1][s1_2] != 1 or gcd_table[s1_1][s2_2] != 1 or
                    gcd_table[s1_1][s3_2] != 1 or gcd_table[s2_1][s1_2] != 1 or
                    gcd_table[s2_1][s2_2] != 1 or gcd_table[s2_1][s3_2] != 1 or
                    gcd_table[s3_1][s1_2] != 1 or gcd_table[s3_1][s2_2] != 1 or
                    gcd_table[s3_1][s3_2] != 1):
                    continue
                for k in range(j + 1, n):
                    b3, s1_3, s2_3, s3_3 = valid_b[k]
                    if (gcd_table[s1_1][s1_3] != 1 or gcd_table[s1_1][s2_3] != 1 or
                        gcd_table[s1_1][s3_3] != 1 or gcd_table[s2_1][s1_3] != 1 or
                        gcd_table[s2_1][s2_3] != 1 or gcd_table[s2_1][s3_3] != 1 or
                        gcd_table[s3_1][s1_3] != 1 or gcd_table[s3_1][s2_3] != 1 or
                        gcd_table[s3_1][s3_3] != 1 or
                        gcd_table[s1_2][s1_3] != 1 or gcd_table[s1_2][s2_3] != 1 or
                        gcd_table[s1_2][s3_3] != 1 or gcd_table[s2_2][s1_3] != 1 or
                        gcd_table[s2_2][s2_3] != 1 or gcd_table[s2_2][s3_3] != 1 or
                        gcd_table[s3_2][s1_3] != 1 or gcd_table[s3_2][s2_3] != 1 or
                        gcd_table[s3_2][s3_3] != 1):
                        continue
                    all_sums = [s1_1, s2_1, s3_1, s1_2, s2_2, s3_2, s1_3, s2_3, s3_3]
                    distinct_count = len(set(all_sums))
                    if distinct_count > best_score:
                        best_score = distinct_count
                        best_result = [list(A), [b1, b2, b3]]
                        if best_score == 9:
                            return best_result
    
    return best_result
# EVOLVE-BLOCK-END
