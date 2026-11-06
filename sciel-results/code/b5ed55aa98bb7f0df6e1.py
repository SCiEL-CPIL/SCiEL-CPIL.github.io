"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin before 90s limit
    TARGET = 123457

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Generate primes up to a reasonable bound
    primes_all = sieve(10000)

    def largest_prime_factor(n):
        """Compute the largest prime factor of n via trial division."""
        if n <= 1:
            return 1
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp

    def generate_smooth_numbers_set(primes, limit):
        """
        Generate all numbers <= limit whose prime factors are all in primes.
        Returns a set of such numbers.
        Uses iterative BFS/DFS approach.
        """
        if not primes:
            return {1}
        
        smooth = {1}
        for p in primes:
            new_smooth = set(smooth)
            # Multiply existing numbers by powers of p
            for x in list(smooth):
                val = x
                while val * p <= limit:
                    val *= p
                    new_smooth.add(val)
            smooth = new_smooth
        return smooth

    # Initial best: [1, TARGET-1]
    best_result = [1, TARGET - 1]
    best_score = largest_prime_factor(TARGET - 1)  # P(123456)
    
    # Also try [TARGET-1, 1]
    # But since we want to minimize max(P(a), P(b)), and P(1)=1,
    # the score for [1, TARGET-1] is max(1, P(TARGET-1)) = P(TARGET-1)

    # Try increasing smoothness bounds: use first k primes
    for k in range(1, len(primes_all) + 1):
        if time.time() > deadline - 5:
            break

        primes_subset = primes_all[:k]
        max_p = primes_subset[-1]

        # If max_p is already >= best_score, we might still find a better pair
        # where both have smaller LPF, but let's continue searching

        # Generate smooth numbers as a set for O(1) lookup
        smooth_set = generate_smooth_numbers_set(primes_subset, TARGET)
        
        if len(smooth_set) < 2:
            continue

        # Check all pairs: for each a in smooth_set, check if TARGET-a is also in smooth_set
        found_at_this_level = False
        min_score_at_level = float('inf')
        best_pair_at_level = None

        for a in smooth_set:
            if time.time() > deadline - 3:
                break
            b = TARGET - a
            if b < 1:
                continue
            if b in smooth_set:
                # Both a and b are max_p-smooth
                # Compute actual P(a) and P(b)
                pa = largest_prime_factor(a)
                pb = largest_prime_factor(b)
                score = max(pa, pb)
                if score < min_score_at_level:
                    min_score_at_level = score
                    best_pair_at_level = (a, b)
                found_at_this_level = True

        if found_at_this_level:
            if min_score_at_level < best_score:
                best_score = min_score_at_level
                best_result = list(best_pair_at_level)
            
            # Since we're going in increasing k, and we found a pair at this level,
            # the optimal score is at most max_p. Any better pair would have been
            # found at an earlier level (with smaller primes).
            # Therefore, we can break as soon as we find a pair.
            break

    return best_result

# EVOLVE-BLOCK-END
