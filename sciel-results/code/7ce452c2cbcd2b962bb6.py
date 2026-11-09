"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'n': 256}

def _is_prime(n):
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

def _can_add(x, S):
    """Check if x can be added to set S maintaining the no-triple-divisibility property.
    For every triple of distinct members (a,b,c), a must not divide b*c.
    When adding x, we need to check:
    1. x does not divide b*c for any distinct b,c in S
    2. For any distinct a,c in S: a does not divide x*c, and c does not divide x*a
    """
    n = len(S)
    # Check x as the "dividing" member: x ∤ S[i]*S[j] for all i<j
    for i in range(n):
        for j in range(i + 1, n):
            if (S[i] * S[j]) % x == 0:
                return False
    # Check x as a "multiplied" member: for distinct a,c in S, a ∤ x*c and c ∤ x*a
    for i in range(n):
        for j in range(i + 1, n):
            if (x * S[j]) % S[i] == 0:
                return False
            if (x * S[i]) % S[j] == 0:
                return False
    return True

def _count_violations_with_x(x, S):
    """Count how many triples would be violated if x were in S. 0 means safe."""
    n = len(S)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (S[i] * S[j]) % x == 0:
                count += 1
            if (x * S[j]) % S[i] == 0:
                count += 1
            if (x * S[i]) % S[j] == 0:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n_max = P['n']
    all_nums = list(range(2, n_max + 1))
    primes = [x for x in all_nums if _is_prime(x)]
    composites = [x for x in all_nums if not _is_prime(x)]

    # Start with all primes (always valid: no prime divides product of two other distinct primes)
    S = list(primes)
    S_set = set(S)

    # Greedy: try to add composites, sorted by number of "conflicts" ascending
    # Fewer conflicts means easier to add
    def conflict_score(x):
        return _count_violations_with_x(x, S)

    # Sort composites by conflict score (recompute as S grows is expensive; use initial as heuristic)
    # Use a simpler heuristic: numbers with fewer small prime factors are safer
    def num_distinct_prime_factors(x):
        cnt = 0
        d = 2
        tmp = x
        while d * d <= tmp:
            if tmp % d == 0:
                cnt += 1
                while tmp % d == 0:
                    tmp //= d
            d += 1
        if tmp > 1:
            cnt += 1
        return cnt

    composites.sort(key=lambda x: (num_distinct_prime_factors(x), x))

    # Greedy addition phase
    for x in composites:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(x, S):
            S.append(x)
            S_set.add(x)

    best_S = list(S)
    best_len = len(S)

    # Local search: try swaps to improve
    # For each element in S, try removing it and adding a composite not in S
    remaining = [x for x in composites if x not in S_set]
    remaining.sort(key=lambda x: (num_distinct_prime_factors(x), x))

    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for idx in range(len(S)):
            if time.time() > deadline:
                break
            removed = S[idx]
            S_without = S[:idx] + S[idx+1:]
            S_without_set = S_set - {removed}

            # Try adding each remaining element
            for x in remaining:
                if time.time() > deadline:
                    break
                if x in S_without_set:
                    continue
                if _can_add(x, S_without):
                    S_new = S_without + [x]
                    if len(S_new) > best_len:
                        best_S = list(S_new)
                        best_len = len(S_new)
                        S = S_new
                        S_set = S_without_set | {x}
                        # Remove x from remaining
                        try:
                            remaining.remove(x)
                        except ValueError:
                            pass
                        improved = True
                        break
            # Restore S for next iteration
            if not improved:
                S = S_without + [removed]
                S_set = S_without_set | {removed}

    # Final: ensure we return the best found
    if len(best_S) > len(S):
        S = best_S

    # Convert to plain Python ints and return as list
    return [int(x) for x in S]
# EVOLVE-BLOCK-END
