import math
import itertools
import time

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

    # Precompute coprime table for sums up to 256
    max_sum = 256
    coprime_table = [[False] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(2, max_sum + 1):
        for j in range(i, max_sum + 1):
            if math.gcd(i, j) == 1:
                coprime_table[i][j] = True
                coprime_table[j][i] = True

    def fast_pairwise_coprime(sums_set):
        distinct = list(sums_set)
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if not coprime_table[distinct[i]][distinct[j]]:
                    return False
        return True

    # Strategy: fix B to small prime sets, search for A such that all sums are pairwise coprime
    # The key insight: if all 9 sums are distinct primes, they're automatically pairwise coprime.
    # Even if not all are prime, pairwise coprime is easier to check.

    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]

    # Phase 1: Try B = small primes and search for A
    small_prime_sets = [
        (2, 3, 5),
        (2, 3, 7),
        (2, 5, 7),
        (3, 5, 7),
        (2, 3, 11),
        (2, 5, 11),
        (3, 5, 11),
        (2, 3, 13),
        (2, 5, 13),
        (3, 5, 13),
        (5, 7, 11),
        (5, 7, 13),
        (7, 11, 13),
        (2, 7, 11),
        (3, 7, 11),
        (2, 11, 13),
        (3, 11, 13),
    ]

    for B in small_prime_sets:
        if time.time() > deadline - 20:
            break
        b1, b2, b3 = B
        # Search for A = (a1, a2, a3) with distinct values in [1, 128]
        for a1 in range(1, 129):
            if time.time() > deadline - 20:
                break
            for a2 in range(a1 + 1, 129):
                if time.time() > deadline - 20:
                    break
                for a3 in range(a2 + 1, 129):
                    if time.time() > deadline - 20:
                        break
                    sums = [a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3]
                    sums_set = set(sums)
                    count = len(sums_set)
                    if count <= best_count:
                        continue
                    if fast_pairwise_coprime(sums_set):
                        best_count = count
                        best_A = [a1, a2, a3]
                        best_B = list(B)
                        if best_count == 9:
                            return best_A, best_B

    # Phase 2: Broader search - try more B values
    if time.time() < deadline - 15:
        # Try B values that are small integers
        for b1 in range(1, 15):
            if time.time() > deadline - 15:
                break
            for b2 in range(b1 + 1, 15):
                if time.time() > deadline - 15:
                    break
                for b3 in range(b2 + 1, 15):
                    if time.time() > deadline - 15:
                        break
                    B = (b1, b2, b3)
                    for a1 in range(1, 60):
                        if time.time() > deadline - 15:
                            break
                        for a2 in range(a1 + 1, 60):
                            if time.time() > deadline - 15:
                                break
                            for a3 in range(a2 + 1, 60):
                                if time.time() > deadline - 15:
                                    break
                                sums = [a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3]
                                sums_set = set(sums)
                                count = len(sums_set)
                                if count <= best_count:
                                    continue
                                if fast_pairwise_coprime(sums_set):
                                    best_count = count
                                    best_A = [a1, a2, a3]
                                    best_B = list(B)
                                    if best_count == 9:
                                        return best_A, best_B

    # Phase 3: Random search with full range
    import random
    random.seed(42)
    for _ in range(200000):
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
        if fast_pairwise_coprime(sums_set):
            best_count = count
            best_A = A
            best_B = B
            if best_count == 9:
                return best_A, best_B

    # Phase 4: Local search - try to improve current best
    if time.time() < deadline - 3:
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
                    if count > best_count and fast_pairwise_coprime(sums_set):
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
                    if count > best_count and fast_pairwise_coprime(sums_set):
                        best_B = new_B
                        best_count = count
                        improved = True
                        if best_count == 9:
                            return best_A, best_B
                        break

    return best_A, best_B
