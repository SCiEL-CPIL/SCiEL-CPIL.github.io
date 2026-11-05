import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

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

    odd_vals = [x for x in range(1, 129) if x % 2 == 1]
    even_vals = [x for x in range(2, 129) if x % 2 == 0]

    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    best_count = 0

    # Phase 1: Systematic search with small values
    # Try odd values up to 40, even values up to 40
    max_odd = 40
    max_even = 40

    odds_small = [x for x in odd_vals if x <= max_odd]
    evens_small = [x for x in even_vals if x <= max_even]

    A_triples = list(combinations(odds_small, 3))
    B_triples = list(combinations(evens_small, 3))

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

    # Phase 2: Expand search range
    if best_count < 9 and time.time() < deadline - 10:
        max_odd = 80
        max_even = 80
        odds_med = [x for x in odd_vals if x <= max_odd]
        evens_med = [x for x in even_vals if x <= max_even]
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

    # Phase 3: Also try A=even, B=odd
    if best_count < 9 and time.time() < deadline - 10:
        A_triples3 = list(combinations(evens_small, 3))
        B_triples3 = list(combinations(odds_small, 3))
        for A in A_triples3:
            if time.time() > deadline:
                break
            a1, a2, a3 = A
            for B in B_triples3:
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

    # Phase 4: Random sampling with larger ranges
    if best_count < 9 and time.time() < deadline - 5:
        import random
        random.seed(42)
        odds_large = [x for x in odd_vals if x <= 128]
        evens_large = [x for x in even_vals if x <= 128]

        for _ in range(5000000):
            if time.time() > deadline:
                break
            A = sorted(random.sample(odds_large, 3))
            B = sorted(random.sample(evens_large, 3))
            s = tuple(a + b for a in A for b in B)
            sums_set = set(s)
            count = len(sums_set)
            if count <= best_count:
                continue
            if fast_pairwise_coprime(sums_set):
                best_count = count
                best_A = A
                best_B = B
                if best_count == 9:
                    return [best_A, best_B]

    return [best_A, best_B]
