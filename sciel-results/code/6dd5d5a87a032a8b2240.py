import time
import math
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    # All distinct sums must be pairwise coprime.
    # If all 9 sums are distinct odd primes, they are automatically pairwise coprime.
    # Strategy: A = {a1, a2, a3} all odd, B = {b1, b2, b3} all even.
    # Sums: a_i + b_j for i,j in {1,2,3} -> 9 sums, all odd.
    # We need all 9 sums to be distinct primes.

    # Precompute primes in [3, 255]
    primes_set = set()
    for n in range(3, 256):
        if n % 2 == 1 and is_prime(n):
            primes_set.add(n)

    def compute_score(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        slist = sorted(sums)
        for i in range(len(slist)):
            for j in range(i+1, len(slist)):
                if math.gcd(slist[i], slist[j]) != 1:
                    return 0
        return len(slist)

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = compute_score(best_A, best_B)

    # Search over difference patterns:
    # e1, e2: even differences for A (a2-a1, a3-a1)
    # d1, d2: even differences for B (b2-b1, b3-b1)
    # base: odd, = a1 + b1
    # All 9 sums = base + e_i + d_j must be distinct odd primes

    max_diff = 80

    for e1 in range(2, max_diff + 1, 2):
        if time.time() > deadline:
            break
        for e2 in range(2, max_diff + 1, 2):
            if e2 == e1:
                continue
            if time.time() > deadline:
                break
            for d1 in range(2, max_diff + 1, 2):
                if time.time() > deadline:
                    break
                for d2 in range(2, max_diff + 1, 2):
                    if d2 == d1:
                        continue
                    if time.time() > deadline:
                        break

                    offsets = [(0,0), (0,d1), (0,d2), (e1,0), (e1,d1), (e1,d2),
                               (e2,0), (e2,d1), (e2,d2)]

                    for base in range(3, 256, 2):
                        if time.time() > deadline:
                            break

                        all_prime = True
                        for (ei, dj) in offsets:
                            s = base + ei + dj
                            if s < 2 or s > 255 or s not in primes_set:
                                all_prime = False
                                break
                        if not all_prime:
                            continue

                        # Find valid a1, b1 decomposition
                        a1_min = max(1, base - 128)
                        a1_max = min(127, base - 2)
                        if a1_min % 2 == 0:
                            a1_min += 1
                        if a1_max % 2 == 0:
                            a1_max -= 1

                        for a1 in range(a1_min, a1_max + 1, 2):
                            b1 = base - a1
                            a2 = a1 + e1
                            a3 = a1 + e2
                            b2 = b1 + d1
                            b3 = b1 + d2

                            if not (1 <= a2 <= 127 and 1 <= a3 <= 127):
                                continue
                            if not (2 <= b2 <= 128 and 2 <= b3 <= 128):
                                continue
                            if len({a1, a2, a3}) != 3:
                                continue
                            if len({b1, b2, b3}) != 3:
                                continue

                            A = [a1, a2, a3]
                            B = [b1, b2, b3]
                            return [A, B]

    return [best_A, best_B]

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import time
    import math

    start_time = time.time()
    deadline = start_time + 85

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes_set = set()
    for n in range(3, 256):
        if n % 2 == 1 and is_prime(n):
            primes_set.add(n)

    def compute_score(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        slist = sorted(sums)
        for i in range(len(slist)):
            for j in range(i+1, len(slist)):
                if math.gcd(slist[i], slist[j]) != 1:
                    return 0
        return len(slist)

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = compute_score(best_A, best_B)

    max_diff = 80

    for e1 in range(2, max_diff + 1, 2):
        if time.time() > deadline:
            break
        for e2 in range(2, max_diff + 1, 2):
            if e2 == e1:
                continue
            if time.time() > deadline:
                break
            for d1 in range(2, max_diff + 1, 2):
                if time.time() > deadline:
                    break
                for d2 in range(2, max_diff + 1, 2):
                    if d2 == d1:
                        continue
                    if time.time() > deadline:
                        break

                    offsets = [(0,0), (0,d1), (0,d2), (e1,0), (e1,d1), (e1,d2),
                               (e2,0), (e2,d1), (e2,d2)]

                    for base in range(3, 256, 2):
                        if time.time() > deadline:
                            break

                        all_prime = True
                        for (ei, dj) in offsets:
                            s = base + ei + dj
                            if s < 2 or s > 255 or s not in primes_set:
                                all_prime = False
                                break
                        if not all_prime:
                            continue

                        a1_min = max(1, base - 128)
                        a1_max = min(127, base - 2)
                        if a1_min % 2 == 0:
                            a1_min += 1
                        if a1_max % 2 == 0:
                            a1_max -= 1

                        for a1 in range(a1_min, a1_max + 1, 2):
                            b1 = base - a1
                            a2 = a1 + e1
                            a3 = a1 + e2
                            b2 = b1 + d1
                            b3 = b1 + d2

                            if not (1 <= a2 <= 127 and 1 <= a3 <= 127):
                                continue
                            if not (2 <= b2 <= 128 and 2 <= b3 <= 128):
                                continue
                            if len({a1, a2, a3}) != 3:
                                continue
                            if len({b1, b2, b3}) != 3:
                                continue

                            A = [a1, a2, a3]
                            B = [b1, b2, b3]
                            return [A, B]

    return [best_A, best_B]

# EVOLVE-BLOCK-END
