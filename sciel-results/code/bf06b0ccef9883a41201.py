import time
from math import gcd, isqrt

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 85  # safety margin under 90s wall-clock

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, isqrt(n) + 1, 2):
            if n % i == 0:
                return False
        return True

    def pairwise_coprime(vals):
        v = list(set(vals))
        n = len(v)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(v[i], v[j]) != 1:
                    return False
        return True

    # Precompute primes up to 256
    primes_set = set()
    for n in range(2, 257):
        if is_prime(n):
            primes_set.add(n)

    best_card = 5
    best_AB = ([1, 7, 13], [4, 10, 16])

    def try_pair(A, B):
        nonlocal best_card, best_AB
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) < 2:
            return
        if not pairwise_coprime(sums):
            return
        if len(sums) > best_card:
            best_card = len(sums)
            best_AB = (list(A), list(B))

    # Phase 1: Search for all-9-prime solutions (cardinality 9)
    # For A=[a1,a2,a3], B=[b1,b2,b3], all 9 sums must be prime
    # Strategy: fix A, then for each b1, determine constraints on b2,b3
    # a1+b1, a2+b1, a3+b1 must be prime -> b1 must make these prime
    # Similarly for b2, b3

    # Use a more targeted approach: iterate over small A values
    # where the differences create good prime patterns
    for a1 in range(1, 40):
        if time.time() > DEADLINE:
            break
        for a2 in range(a1 + 1, 40):
            if time.time() > DEADLINE:
                break
            for a3 in range(a2 + 1, 40):
                if time.time() > DEADLINE:
                    break
                A = [a1, a2, a3]
                # For each candidate b1, check if a1+b1, a2+b1, a3+b1 are all prime
                # Then for b2, check a1+b2, a2+b2, a3+b2 are all prime
                # And they must be distinct from the b1 sums (or at least pairwise coprime)
                valid_b1 = []
                for b1 in range(1, 129):
                    s1 = a1 + b1
                    s2 = a2 + b1
                    s3 = a3 + b1
                    if s1 in primes_set and s2 in primes_set and s3 in primes_set:
                        valid_b1.append(b1)

                if len(valid_b1) < 2:
                    continue

                valid_b2 = []
                for b2 in range(1, 129):
                    if b2 in valid_b1:  # must be distinct
                        continue
                    s1 = a1 + b2
                    s2 = a2 + b2
                    s3 = a3 + b2
                    if s1 in primes_set and s2 in primes_set and s3 in primes_set:
                        valid_b2.append(b2)

                if len(valid_b2) < 2:
                    continue

                # Now find three distinct values from valid_b1 + valid_b2
                all_valid_b = set(valid_b1) | set(valid_b2)
                all_valid_b = [b for b in all_valid_b if 1 <= b <= 128]

                if len(all_valid_b) < 3:
                    continue

                # Try combinations of 3 distinct b values
                # But first, check if we can get all 9 sums prime
                # We need 3 b's such that for each b, all 3 sums with A are prime
                # AND all 9 sums are distinct (for cardinality 9) or at least pairwise coprime

                # Efficient: pick 3 from all_valid_b
                n_b = len(all_valid_b)
                if n_b <= 50:  # manageable
                    for i in range(n_b):
                        if time.time() > DEADLINE:
                            break
                        for j in range(i + 1, n_b):
                            if time.time() > DEADLINE:
                                break
                            for k in range(j + 1, n_b):
                                if time.time() > DEADLINE:
                                    break
                                B = [all_valid_b[i], all_valid_b[j], all_valid_b[k]]
                                # Check all 9 sums are prime
                                all_prime = True
                                for a in A:
                                    for b in B:
                                        if (a + b) not in primes_set:
                                            all_prime = False
                                            break
                                    if not all_prime:
                                        break
                                if all_prime:
                                    return A, B
                                # Even if not all prime, check pairwise coprime
                                try_pair(A, B)
                                if best_card == 9:
                                    return A, B

    # Phase 2: Broader search for high-cardinality pairwise-coprime solutions
    # Use structured A with small values
    for a1 in range(1, 25):
        if time.time() > DEADLINE:
            break
        for a2 in range(a1 + 1, 25):
            if time.time() > DEADLINE:
                break
            for a3 in range(a2 + 1, 25):
                if time.time() > DEADLINE:
                    break
                A = [a1, a2, a3]
                # Search B in a wider range
                for b1 in range(1, 60):
                    if time.time() > DEADLINE:
                        break
                    for b2 in range(b1 + 1, 60):
                        if time.time() > DEADLINE:
                            break
                        for b3 in range(b2 + 1, 60):
                            if time.time() > DEADLINE:
                                break
                            B = [b1, b2, b3]
                            try_pair(A, B)
                            if best_card == 9:
                                return A, B

    return list(best_AB[0]), list(best_AB[1])
# EVOLVE-BLOCK-END
