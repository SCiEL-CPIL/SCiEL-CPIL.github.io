import math
import itertools
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def pairwise_coprime(lst):
        n = len(lst)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    # Precompute coprime compatibility matrix for values 2..256
    # coprime[x][y] = True if gcd(x,y)==1
    MAX_SUM = 256
    coprime = [[False] * (MAX_SUM + 1) for _ in range(MAX_SUM + 1)]
    for i in range(2, MAX_SUM + 1):
        for j in range(i + 1, MAX_SUM + 1):
            if gcd(i, j) == 1:
                coprime[i][j] = True
                coprime[j][i] = True

    def is_pairwise_coprime_fast(sums_list):
        """Check if all distinct values in sums_list are pairwise coprime using precomputed matrix."""
        distinct = list(set(sums_list))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if not coprime[distinct[i]][distinct[j]]:
                    return False
        return True

    # Verify seed
    seed_A = [1, 7, 13]
    seed_B = [4, 10, 16]
    seed_sums = [a + b for a in seed_A for b in seed_B]
    if is_pairwise_coprime_fast(seed_sums):
        best_A = seed_A
        best_B = seed_B
        best_count = len(set(seed_sums))
    else:
        best_A = [1, 2, 3]
        best_B = [1, 2, 3]
        best_count = 0

    # Precompute primes up to 256
    primes = set()
    for n in range(2, 257):
        is_p = True
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                is_p = False
                break
        if is_p:
            primes.add(n)

    # Strategy: Search over parameterized 3x3 grids
    # A = [a, a+d1, a+d1+d2], B = [b, b+e1, b+e1+e2]
    # Sums form: c + di + ej where c=a+b, di in {0,d1,d1+d2}, ej in {0,e1,e1+e2}
    # For 9 distinct sums: all 9 values c+di+ej must be distinct
    # For pairwise coprime: all distinct sums must be pairwise coprime

    # Phase 1: Try to find 9 distinct pairwise coprime sums (all prime is sufficient)
    # c = a+b, a>=1, b>=1 => c>=2
    # a+d1+d2 <= 128, b+e1+e2 <= 128
    # c + d1 + d2 + e1 + e2 <= 256

    found_9 = False
    max_delta = 35

    for c in range(2, 200):
        if time.time() > deadline - 10:
            break
        max_d_sum = min(max_delta, 256 - c - 2)  # leave room for e1+e2 >= 2
        for d1 in range(1, max_d_sum):
            if time.time() > deadline - 10:
                break
            for d2 in range(1, max_d_sum - d1 + 1):
                d_sum = d1 + d2
                if c + d_sum + 2 > 256:
                    break
                max_e_sum = min(max_delta, 256 - c - d_sum)
                for e1 in range(1, max_e_sum):
                    if time.time() > deadline - 10:
                        break
                    for e2 in range(1, max_e_sum - e1 + 1):
                        e_sum = e1 + e2
                        if c + d_sum + e_sum > 256:
                            break

                        # Compute the 9 sums
                        s00 = c
                        s01 = c + e1
                        s02 = c + e_sum
                        s10 = c + d1
                        s11 = c + d1 + e1
                        s12 = c + d1 + e_sum
                        s20 = c + d_sum
                        s21 = c + d_sum + e1
                        s22 = c + d_sum + e_sum

                        sums9 = [s00, s01, s02, s10, s11, s12, s20, s21, s22]

                        # Check all distinct
                        if len(set(sums9)) != 9:
                            continue

                        # Check pairwise coprime using fast method
                        if not is_pairwise_coprime_fast(sums9):
                            continue

                        # Check feasibility: find a, b such that
                        # a = c - b, 1 <= a <= 128, 1 <= b <= 128
                        # a + d_sum = c - b + d_sum <= 128 => b >= c + d_sum - 128
                        # b + e_sum <= 128 => b <= 128 - e_sum
                        # a >= 1 => b <= c - 1
                        # b >= 1
                        b_min = max(1, c + d_sum - 128)
                        b_max = min(c - 1, 128 - e_sum)

                        if b_min <= b_max:
                            b = b_min
                            a = c - b
                            A = [a, a + d1, a + d_sum]
                            B = [b, b + e1, b + e_sum]

                            if all(1 <= x <= 128 for x in A + B) and len(set(A)) == 3 and len(set(B)) == 3:
                                best_A = A
                                best_B = B
                                best_count = 9
                                found_9 = True
                                return [A, B]

    # Phase 2: Try to find 8 or 7 distinct pairwise coprime sums
    # Allow some sum collisions but ensure remaining distinct sums are pairwise coprime
    if time.time() < deadline - 5:
        # Search with slightly different strategy: try to maximize distinct count
        # while maintaining pairwise coprimality
        for c in range(2, 180):
            if time.time() > deadline - 3:
                break
            for d1 in range(1, 30):
                if time.time() > deadline - 3:
                    break
                for d2 in range(1, 30):
                    d_sum = d1 + d2
                    if c + d_sum + 2 > 256:
                        break
                    for e1 in range(1, 30):
                        if time.time() > deadline - 3:
                            break
                        for e2 in range(1, 30):
                            e_sum = e1 + e2
                            if c + d_sum + e_sum > 256:
                                break

                            s00 = c
                            s01 = c + e1
                            s02 = c + e_sum
                            s10 = c + d1
                            s11 = c + d1 + e1
                            s12 = c + d1 + e_sum
                            s20 = c + d_sum
                            s21 = c + d_sum + e1
                            s22 = c + d_sum + e_sum

                            sums9 = [s00, s01, s02, s10, s11, s12, s20, s21, s22]
                            distinct_sums = list(set(sums9))
                            n_distinct = len(distinct_sums)

                            if n_distinct <= best_count:
                                continue

                            if not is_pairwise_coprime_fast(distinct_sums):
                                continue

                            b_min = max(1, c + d_sum - 128)
                            b_max = min(c - 1, 128 - e_sum)

                            if b_min <= b_max:
                                b = b_min
                                a = c - b
                                A = [a, a + d1, a + d_sum]
                                B = [b, b + e1, b + e_sum]

                                if all(1 <= x <= 128 for x in A + B) and len(set(A)) == 3 and len(set(B)) == 3:
                                    all_sums = [x + y for x in A for y in B]
                                    ds = list(set(all_sums))
                                    if len(ds) > best_count and is_pairwise_coprime_fast(ds):
                                        best_A = A
                                        best_B = B
                                        best_count = len(ds)

    return [best_A, best_B]
