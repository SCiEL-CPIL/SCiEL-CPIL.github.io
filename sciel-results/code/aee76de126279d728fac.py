import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    # Precompute gcd for values up to 256
    max_sum = 256
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1):
            gcd_table[i][j] = math.gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    def fast_pairwise_coprime(sums_set):
        """Fast check using precomputed gcd table."""
        distinct = list(sums_set)
        n = len(distinct)
        for i in range(n):
            vi = distinct[i]
            for j in range(i + 1, n):
                vj = distinct[j]
                if gcd_table[vi][vj] != 1:
                    return False
        return True

    # Search strategy:
    # To maximize cardinality (9), we need all sums to be distinct.
    # To satisfy pairwise coprime, we prefer sums to be primes or 1 (but sums >= 3).
    # Odd A + Even B -> Odd Sums. This avoids factor 2 issues.
    
    odds = [x for x in range(1, 129, 2)]
    evens = [x for x in range(2, 129, 2)]
    
    # Limit search space to smaller numbers for speed, as they are more likely to be coprime?
    # Actually, larger numbers might be safer from small prime factors, but distinctness is harder?
    # Let's try a moderate range first.
    
    max_val = 60
    odds_small = [x for x in odds if x <= max_val]
    evens_small = [x for x in evens if x <= max_val]
    
    A_triples = list(combinations(odds_small, 3))
    B_triples = list(combinations(evens_small, 3))
    
    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    best_count = 0

    for A in A_triples:
        if time.time() > deadline:
            break
        a1, a2, a3 = A
        for B in B_triples:
            if time.time() > deadline:
                break
            b1, b2, b3 = B
            s = (a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3)
            sums_set = set(s)
            count = len(sums_set)
            if count <= best_count:
                continue
            if fast_pairwise_coprime(sums_set):
                best_count = count
                best_A = list(A)
                best_B = list(B)
                if best_count == 9:
                    return [best_A, best_B]

    # If not found in small range, expand
    if best_count < 9 and time.time() < deadline - 10:
        max_val = 100
        odds_med = [x for x in odds if x <= max_val]
        evens_med = [x for x in evens if x <= max_val]
        A_triples2 = list(combinations(odds_med, 3))
        B_triples2 = list(combinations(evens_med, 3))

        for A in A_triples2:
            if time.time() > deadline:
                break
            a1, a2, a3 = A
            for B in B_triples2:
                if time.time() > deadline:
                    break
                b1, b2, b3 = B
                s = (a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3)
                sums_set = set(s)
                count = len(sums_set)
                if count <= best_count:
                    continue
                if fast_pairwise_coprime(sums_set):
                    best_count = count
                    best_A = list(A)
                    best_B = list(B)
                    if best_count == 9:
                        return [best_A, best_B]

    return [best_A, best_B]
