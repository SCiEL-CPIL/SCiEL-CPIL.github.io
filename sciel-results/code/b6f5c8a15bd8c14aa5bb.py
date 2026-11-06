"""Program: find A, B (3 distinct ints in [1,128]) maximizing |A+B| with pairwise-coprime distinct sums."""
import itertools
import math
import time
import random
import sys

# EVOLVE-BLOCK-START
def _sieve(n):
    is_prime = [True] * (n + 1)
    if n >= 0:
        is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:n + 1:step] = [False] * len(range(start, n + 1, step))
    return is_prime

def construct():
    t_start = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    MAX_SUM = 256
    is_prime = _sieve(MAX_SUM)

    def all_prime(lst):
        for x in lst:
            if x < 2 or not is_prime[x]:
                return False
        return True

    # We parameterize A = {a, a+d1, a+2*d1}, B = {b, b+d2, b+2*d2}
    # Sums = {a+b + (i*d1 + j*d2) : i,j in {0,1,2}}
    # If gcd(d1, d2) == 1, the 9 sums are distinct.
    # We want all 9 to be primes (hence pairwise coprime).
    #
    # Constraints:
    #   a >= 1, a+2*d1 <= 128
    #   b >= 1, b+2*d2 <= 128
    #   a, a+d1, a+2*d1 distinct (d1 >= 1)
    #   b, b+d2, b+2*d2 distinct (d2 >= 1)
    #
    # Strategy: enumerate (d1, d2) with gcd(d1,d2)==1, then for each (a, b),
    # check if all 9 sums are prime.
    #
    # To speed up: for a fixed (d1, d2, a), the sums as a function of b are:
    #   s_ij(b) = (a + b) + i*d1 + j*d2
    # All 9 must be prime. We can precompute which b values work.
    #
    # Even faster: for fixed (d1, d2), iterate a and b and check.
    # Number of (d1, d2) pairs with gcd=1 and 2*d1 <= 127, 2*d2 <= 127: manageable.
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5  # seed score

    def score_of(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        n = len(distinct)
        # Check pairwise coprime
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) > 1:
                    return 0
        return n

    # Precompute current best
    best_score = score_of(best_A, best_B)

    # Enumerate (d1, d2) pairs
    max_d = 63  # since 2*d <= 127
    found_perfect = False

    for d1 in range(1, max_d + 1):
        if time.time() - t_start > DEADLINE:
            break
        for d2 in range(1, max_d + 1):
            if math.gcd(d1, d2) != 1:
                continue
            if (d1 + d2) % 100 == 0 and time.time() - t_start > DEADLINE:
                break
            
            # a ranges from 1 to 128 - 2*d1
            # b ranges from 1 to 128 - 2*d2
            a_max = 128 - 2 * d1
            b_max = 128 - 2 * d2
            if a_max < 1 or b_max < 1:
                continue
            
            # For each (a, b), check if all 9 sums are prime
            # Sum offsets: i*d1 + j*d2 for i,j in {0,1,2}
            offsets = [i * d1 + j * d2 for i in range(3) for j in range(3)]
            
            for a in range(1, a_max + 1):
                if (a % 100 == 0) and time.time() - t_start > DEADLINE:
                    break
                base_a = a
                for b in range(1, b_max + 1):
                    base = base_a + b
                    # Check all 9 sums are prime
                    ok = True
                    for off in offsets:
                        s = base + off
                        if s > MAX_SUM or not is_prime[s]:
                            ok = False
                            break
                    if ok:
                        A = [a, a + d1, a + 2 * d1]
                        B = [b, b + d2, b + 2 * d2]
                        return [A, B]
    
    # If we didn't find a perfect 9, search for best score < 9
    # This is unlikely to be needed, but for completeness
    # Try a broader search for high-score solutions
    # Actually, let's just try random sampling if time permits
    
    if time.time() - t_start < DEADLINE - 5:
        # Try random A, B and score them
        rng = random.Random(42)
        for _ in range(100000):
            if time.time() - t_start > DEADLINE:
                break
            A = rng.sample(range(1, 129), 3)
            B = rng.sample(range(1, 129), 3)
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A = A
                best_B = B
                if s >= 9:
                    return [A, B]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
