import math
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    def is_pairwise_coprime(vals):
        distinct = list(set(vals))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    best_count = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Phase 1: Exhaustive search over small range (1-40)
    # C(40,3) = 9880, so 9880^2 = ~97M pairs — too many
    # Instead, fix B and iterate A more efficiently
    
    # Use a smarter approach: for each B, precompute which A values work
    # For a fixed B = {b1, b2, b3}, the sums for A = {a1, a2, a3} are:
    # a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    
    # For these to be pairwise coprime, each pair must have gcd 1
    
    # Let's limit to range 1-50 and use early termination
    range_limit = 50
    candidates = list(range(1, range_limit + 1))
    
    # Precompute: for each pair (x, y) in [2, 100], is gcd(x,y) == 1?
    # Max sum = 50+50 = 100, min = 1+1 = 2
    max_sum = 2 * range_limit
    coprime_table = [[False] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(2, max_sum + 1):
        for j in range(i, max_sum + 1):
            if math.gcd(i, j) == 1:
                coprime_table[i][j] = True
                coprime_table[j][i] = True
    
    def fast_pairwise_coprime(sums_set):
        """Fast check using precomputed table."""
        distinct = list(sums_set)
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if not coprime_table[distinct[i]][distinct[j]]:
                    return False
        return True
    
    # Phase 1: Structured search
    for B in itertools.combinations(candidates, 3):
        if time.time() > deadline - 15:
            break
        b1, b2, b3 = B
        for A in itertools.combinations(candidates, 3):
            if time.time() > deadline - 15:
                break
            a1, a2, a3 = A
            s = [a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3]
            sums_set = set(s)
            count = len(sums_set)
            if count <= best_count:
                continue
            if fast_pairwise_coprime(sums_set):
                best_count = count
                best_A = list(A)
                best_B = list(B)
                if best_count == 9:
                    return best_A, best_B
    
    # Phase 2: Random search with full range
    random.seed(999)
    for _ in range(500000):
        if time.time() > deadline - 5:
            break
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        sums_set = set()
        for a in A:
            for b in B:
                sums_set.add(a + b)
        count = len(sums_set)
        if count <= best_count:
            continue
        # Quick filter: if any sum is 1, skip (can't happen since min is 2)
        # Check pairwise coprime
        if is_pairwise_coprime(sums_set):
            best_count = count
            best_A = A
            best_B = B
            if best_count == 9:
                return best_A, best_B
    
    # Phase 3: Local search with optimized swap check
    if time.time() < deadline - 3:
        # Precompute coprime table for full range
        max_sum_full = 256
        coprime_full = [[False] * (max_sum_full + 1) for _ in range(max_sum_full + 1)]
        for i in range(2, max_sum_full + 1):
            for j in range(i, max_sum_full + 1):
                if math.gcd(i, j) == 1:
                    coprime_full[i][j] = True
                    coprime_full[j][i] = True
        
        def fast_check_full(sums_set):
            distinct = list(sums_set)
            n = len(distinct)
            for i in range(n):
                for j in range(i + 1, n):
                    if not coprime_full[distinct[i]][distinct[j]]:
                        return False
            return True
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try swaps in A
            for pos in range(3):
                for new_val in range(1, 129):
                    if new_val in best_A:
                        continue
                    new_A = best_A.copy()
                    new_A[pos] = new_val
                    sums_set = set()
                    for a in new_A:
                        for b in best_B:
                            sums_set.add(a + b)
                    count = len(sums_set)
                    if count > best_count and fast_check_full(sums_set):
                        best_A = new_A
                        best_count = count
                        improved = True
                        if best_count == 9:
                            return best_A, best_B
                        break
            # Try swaps in B
            for pos in range(3):
                for new_val in range(1, 129):
                    if new_val in best_B:
                        continue
                    new_B = best_B.copy()
                    new_B[pos] = new_val
                    sums_set = set()
                    for a in best_A:
                        for b in new_B:
                            sums_set.add(a + b)
                    count = len(sums_set)
                    if count > best_count and fast_check_full(sums_set):
                        best_B = new_B
                        best_count = count
                        improved = True
                        if best_count == 9:
                            return best_A, best_B
                        break
    
    return best_A, best_B
