"""Search-based program: generate varied candidate (A,B) pairs and keep the best valid one."""
import time
import random
from math import gcd
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80

    # Precompute prime factors for numbers up to 256
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    pf_cache = {i: prime_factors(i) for i in range(2, 257)}
    pf_cache[1] = set()

    def coprime(a, b):
        if a == 1 or b == 1:
            return True
        if a in pf_cache and b in pf_cache:
            return len(pf_cache[a] & pf_cache[b]) == 0
        return gcd(a, b) == 1

    def is_pairwise_coprime(vals):
        lst = list(vals)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if not coprime(lst[i], lst[j]):
                    return False
        return True

    def distinct_sums(A, B):
        return {a + b for a in A for b in B}

    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)

    # Initial best
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    if best_score < 0:
        best_score = 0

    # Build pool: primes, 1, and useful composites
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [n for n in range(1, 129) if is_prime(n)]
    composites = [6, 10, 14, 15, 21, 22, 26, 33, 34, 35, 38, 39, 46, 51, 55, 57, 58, 62, 65, 69, 74, 77, 82, 85, 86, 87, 91, 93, 94, 95, 105, 106, 110, 114, 115, 118, 119]
    pool = list(set(primes + composites + [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 17, 19, 23, 25, 27, 29, 31, 37]))
    pool = [x for x in pool if 1 <= x <= 128]
    random.shuffle(pool)

    # Structured candidates (broader list)
    structured = [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 3, 5], [2, 4, 6]),
        ([1, 7, 13], [4, 10, 16]),
        ([2, 3, 5], [7, 11, 13]),
        ([1, 2, 4], [3, 5, 7]),
        ([3, 5, 7], [2, 4, 6]),
        ([1, 4, 9], [2, 3, 5]),
        ([2, 5, 11], [3, 7, 13]),
        ([1, 3, 7], [2, 5, 11]),
        ([4, 9, 16], [2, 3, 5]),
        ([1, 2, 5], [3, 7, 11]),
        ([1, 3, 8], [2, 5, 10]),
        ([2, 4, 9], [1, 5, 7]),
        ([3, 5, 10], [1, 4, 8]),
        ([1, 6, 11], [2, 5, 9]),
        ([2, 7, 12], [1, 4, 8]),
        ([1, 4, 10], [3, 6, 11]),
        ([2, 5, 14], [1, 7, 12]),
        ([1, 8, 15], [2, 5, 11]),
        ([3, 6, 13], [1, 4, 10]),
        ([1, 9, 14], [2, 5, 11]),
        ([4, 7, 12], [1, 5, 9]),
        ([2, 8, 15], [1, 5, 11]),
        ([1, 10, 17], [2, 5, 11]),
        ([3, 7, 14], [1, 4, 10]),
        ([1, 11, 19], [2, 5, 13]),
        ([2, 9, 16], [1, 5, 11]),
        ([4, 8, 15], [1, 5, 11]),
        ([1, 12, 20], [2, 5, 13]),
        ([3, 8, 17], [1, 4, 10]),
        ([1, 5, 11], [2, 7, 13]),
        ([2, 6, 11], [1, 5, 9]),
        ([1, 4, 8], [3, 7, 11]),
        ([2, 5, 9], [1, 4, 8]),
        ([1, 6, 10], [3, 7, 11]),
        ([3, 5, 9], [1, 4, 8]),
        ([1, 7, 11], [2, 5, 9]),
        ([2, 6, 10], [1, 5, 9]),
        ([1, 8, 12], [3, 5, 9]),
        ([2, 7, 11], [1, 4, 8]),
        ([1, 9, 13], [2, 5, 10]),
        ([3, 6, 10], [1, 4, 8]),
        ([1, 5, 13], [2, 6, 10]),
        ([2, 4, 10], [1, 5, 9]),
        ([1, 7, 15], [2, 4, 8]),
        ([3, 5, 11], [1, 4, 8]),
        ([1, 6, 14], [2, 5, 9]),
        ([2, 8, 12], [1, 5, 7]),
        ([1, 9, 15], [2, 4, 8]),
        ([3, 7, 11], [1, 4, 8]),
        ([1, 10, 14], [2, 5, 9]),
        ([2, 5, 13], [1, 6, 10]),
        ([1, 8, 16], [2, 5, 9]),
        ([3, 6, 12], [1, 4, 8]),
        ([1, 11, 15], [2, 4, 8]),
        ([2, 7, 13], [1, 5, 9]),
        ([1, 9, 17], [2, 5, 11]),
        ([3, 8, 14], [1, 4, 10]),
        ([1, 12, 18], [2, 5, 9]),
        ([2, 6, 14], [1, 5, 9]),
    ]
    for A, B in structured:
        if time.time() > deadline:
            break
        s = score_of(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]
        if best_score >= 9:
            break

    # Constraint-aware random search
    def constraint_aware_score(A, B):
        """Score with early termination for speed."""
        sums = set()
        for a in A:
            for b in B:
                s = a + b
                if s in sums:
                    continue
                # Check coprime with all existing sums
                ok = True
                for existing in sums:
                    if not coprime(s, existing):
                        ok = False
                        break
                if not ok:
                    return -1
                sums.add(s)
        if len(sums) < 2:
            return -1
        return len(sums)

    # Random search
    while time.time() < deadline - 5:
        if random.random() < 0.1:
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
        else:
            A = random.sample(pool, 3)
            B = random.sample(pool, 3)
        s = constraint_aware_score(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]
            if best_score >= 9:
                break

    # Local refinement: single-element swaps (exhaustive over pool)
    if best_score < 9 and time.time() < deadline - 3:
        for _ in range(30000):
            if time.time() > deadline - 2:
                break
            A = best_A[:]
            B = best_B[:]
            if random.random() < 0.5:
                idx = random.randrange(3)
                A[idx] = random.choice(pool)
            else:
                idx = random.randrange(3)
                B[idx] = random.choice(pool)
            if len(set(A)) != 3 or len(set(B)) != 3:
                continue
            if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
                continue
            s = constraint_aware_score(A, B)
            if s > best_score:
                best_score = s
                best_A, best_B = A[:], B[:]
                if best_score >= 9:
                    break

    # Two-element swaps
    if best_score < 9 and time.time() < deadline - 2:
        for _ in range(20000):
            if time.time() > deadline - 1.5:
                break
            A = best_A[:]
            B = best_B[:]
            if random.random() < 0.5:
                idx1, idx2 = random.sample(range(3), 2)
                A[idx1] = random.choice(pool)
                A[idx2] = random.choice(pool)
            else:
                idx1, idx2 = random.sample(range(3), 2)
                B[idx1] = random.choice(pool)
                B[idx2] = random.choice(pool)
            if len(set(A)) != 3 or len(set(B)) != 3:
                continue
            if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
                continue
            s = constraint_aware_score(A, B)
            if s > best_score:
                best_score = s
                best_A, best_B = A[:], B[:]
                if best_score >= 9:
                    break

    # Cross swaps (one in A, one in B)
    if best_score < 9 and time.time() < deadline - 1:
        for _ in range(15000):
            if time.time() > deadline - 0.5:
                break
            A = best_A[:]
            B = best_B[:]
            idxA = random.randrange(3)
            idxB = random.randrange(3)
            A[idxA] = random.choice(pool)
            B[idxB] = random.choice(pool)
            if len(set(A)) != 3 or len(set(B)) != 3:
                continue
            if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
                continue
            s = constraint_aware_score(A, B)
            if s > best_score:
                best_score = s
                best_A, best_B = A[:], B[:]
                if best_score >= 9:
                    break

    # Final exhaustive single-swap polish
    if best_score < 9 and time.time() < deadline - 0.5:
        for idx in range(3):
            for val in pool:
                if time.time() > deadline - 0.3:
                    break
                A = best_A[:]
                B = best_B[:]
                A[idx] = val
                if len(set(A)) == 3 and all(1 <= x <= 128 for x in A):
                    s = constraint_aware_score(A, B)
                    if s > best_score:
                        best_score = s
                        best_A = A[:]
                        if best_score >= 9:
                            return [best_A, best_B]
        for idx in range(3):
            for val in pool:
                if time.time() > deadline - 0.3:
                    break
                A = best_A[:]
                B = best_B[:]
                B[idx] = val
                if len(set(B)) == 3 and all(1 <= x <= 128 for x in B):
                    s = constraint_aware_score(A, B)
                    if s > best_score:
                        best_score = s
                        best_B = B[:]
                        if best_score >= 9:
                            return [best_A, best_B]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
