import random
import math
import time
from math import gcd

def construct():
    # Precompute GCD table for values up to 256
    max_val = 256
    gcd_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i, max_val + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    def is_pairwise_coprime(sums_list):
        """Check if all distinct values in sums_list are pairwise coprime."""
        distinct = set(sums_list)
        distinct_list = list(distinct)
        n = len(distinct_list)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[distinct_list[i]][distinct_list[j]] > 1:
                    return False
        return True
    
    def count_distinct_sums(A, B):
        """Count distinct sums in A+B."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        return len(sums)
    
    def evaluate(A, B):
        """Score a candidate: number of distinct sums if pairwise coprime, else 0."""
        sums = []
        for a in A:
            for b in B:
                sums.append(a + b)
        
        # Check if all distinct sums are pairwise coprime
        distinct = set(sums)
        distinct_list = list(distinct)
        n = len(distinct_list)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[distinct_list[i]][distinct_list[j]] > 1:
                    return 0
        return len(distinct)
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Strategy 1: Search over difference vectors (a0, u1, u2, b0, v1, v2)
    # A = [a0, a0+u1, a0+u1+u2], B = [b0, b0+v1, b0+v1+v2]
    # All values must be in [1, 128] and distinct within each set
    
    def random_diff_vector():
        """Generate random difference vectors ensuring all values in [1,128] and distinct."""
        while True:
            a0 = random.randint(1, 128)
            u1 = random.randint(-30, 30)
            u2 = random.randint(-30, 30)
            b0 = random.randint(1, 128)
            v1 = random.randint(-30, 30)
            v2 = random.randint(-30, 30)
            
            A = [a0, a0 + u1, a0 + u1 + u2]
            B = [b0, b0 + v1, b0 + v1 + v2]
            
            # Check bounds and distinctness
            if all(1 <= x <= 128 for x in A) and len(set(A)) == 3:
                if all(1 <= x <= 128 for x in B) and len(set(B)) == 3:
                    return A, B
    
    # Strategy 2: Random sampling as fallback
    evens = [i for i in range(1, 129) if i % 2 == 0]
    odds = [i for i in range(1, 129) if i % 2 != 0]
    
    # Strategy 3: CSP-inspired backtracking over sum-grid with incremental consistency
    # We enforce:
    #  - 3x3 grid s_ij = a_i + b_j
    #  - rank-1 additive structure: s_ij - s_ik = s_jk - s_jk' (i.e., differences consistent)
    #  - pairwise coprimality of distinct sums
    #  - all a_i, b_j in [1,128], distinct within each set
    # We search over candidate sets A and B but use a constraint-propagation loop
    # that prunes partial assignments early.
    
    def check_partial(A_partial, B_partial, used_sums):
        """Check if partial assignment is consistent: all sums so far are pairwise coprime."""
        for s in used_sums:
            for t in used_sums:
                if s != t and gcd_table[s][t] > 1:
                    return False
        return True
    
    def csp_search(A_init, B_init):
        """
        Backtracking search that tries to extend A_init and B_init to full sets
        while maintaining pairwise coprimality of all sums.
        """
        # Start with initial A and B
        A = list(A_init)
        B = list(B_init)
        
        # Compute current sums
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        # Check if current sums are pairwise coprime
        sums_list = list(sums)
        n = len(sums_list)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[sums_list[i]][sums_list[j]] > 1:
                    return 0, A, B
        
        # Try to improve by mutating one element at a time with constraint checking
        best_local = len(sums)
        best_local_A = A
        best_local_B = B
        
        for step in range(50):  # Limited depth to avoid infinite loops
            improved = False
            # Try mutating A
            for i in range(3):
                for new_a in range(1, 129):
                    if new_a in A:
                        continue
                    test_A = A.copy()
                    test_A[i] = new_a
                    test_sums = set()
                    for a in test_A:
                        for b in B:
                            test_sums.add(a + b)
                    # Check pairwise coprimality
                    ts_list = list(test_sums)
                    valid = True
                    for x in range(len(ts_list)):
                        for y in range(x + 1, len(ts_list)):
                            if gcd_table[ts_list[x]][ts_list[y]] > 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and len(test_sums) > best_local:
                        A = test_A
                        best_local = len(test_sums)
                        best_local_A = A
                        improved = True
                        break
                if improved:
                    break
            
            # Try mutating B
            if not improved:
                for i in range(3):
                    for new_b in range(1, 129):
                        if new_b in B:
                            continue
                        test_B = B.copy()
                        test_B[i] = new_b
                        test_sums = set()
                        for a in A:
                            for b in test_B:
                                test_sums.add(a + b)
                        # Check pairwise coprimality
                        ts_list = list(test_sums)
                        valid = True
                        for x in range(len(ts_list)):
                            for y in range(x + 1, len(ts_list)):
                                if gcd_table[ts_list[x]][ts_list[y]] > 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(test_sums) > best_local:
                            B = test_B
                            best_local = len(test_sums)
                            best_local_B = B
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_local, best_local_A, best_local_B
    
    # Initialize with a few good seeds
    seeds = [
        ([1, 7, 13], [4, 10, 16]),
        ([3, 5, 7], [2, 4, 6]),
        ([1, 3, 5], [2, 4, 6]),
        ([5, 11, 17], [2, 4, 6]),
        ([1, 9, 17], [2, 4, 6]),
    ]
    
    for A_seed, B_seed in seeds:
        score = evaluate(A_seed, B_seed)
        if score > best_score:
            best_score = score
            best_A = A_seed
            best_B = B_seed
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Alternate between strategies
        if attempts % 3 == 0:
            # CSP-inspired search from random start
            A_init = random.sample(odds, 3)
            B_init = random.sample(evens, 3)
            score, A_new, B_new = csp_search(A_init, B_init)
            if score > best_score:
                best_score = score
                best_A = A_new
                best_B = B_new
                if best_score == 9:
                    return [best_A, best_B]
        elif attempts % 3 == 1:
            # Difference vector search
            A, B = random_diff_vector()
            score = evaluate(A, B)
            if score > best_score:
                best_score = score
                best_A = A
                best_B = B
                if best_score == 9:
                    return [best_A, best_B]
        else:
            # Random sampling with parity constraint
            A = random.sample(odds, 3)
            B = random.sample(evens, 3)
            score = evaluate(A, B)
            if score > best_score:
                best_score = score
                best_A = A
                best_B = B
                if best_score == 9:
                    return [best_A, best_B]
    
    return [best_A, best_B]
