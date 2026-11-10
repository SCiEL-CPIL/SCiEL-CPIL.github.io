"""Rewritten program: direct construction of a large subset of [1,512] with lcm(a,b) <= 512 for all pairs."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = start + 80.0

    LIMIT = 512

    # Helper: check if a number x can be added to set S
    def can_add(x, S):
        for a in S:
            # lcm(x, a) = x * a // gcd(x, a)
            if x * a // gcd(x, a) > LIMIT:
                return False
        return True

    # Helper: compute the largest prime factor of n
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        # Trial division
        f = 2
        m = n
        while f * f <= m:
            while m % f == 0:
                m //= f
            f += 1
        if m > 1:
            return m
        return f

    # Precompute largest prime factor for all candidates 1..512
    lpf = [1] * (LIMIT + 1)
    for i in range(2, LIMIT + 1):
        lpf[i] = largest_prime_factor(i)

    # Start with the contiguous prefix [1, k] as a strong seed
    def max_contiguous_k(L):
        k = 1
        while True:
            nk = k + 1
            ok = True
            for a in range(1, k + 1):
                if nk * a // gcd(nk, a) > L:
                    ok = False
                    break
            if not ok:
                break
            k = nk
            if time.time() > DEADLINE:
                return k
        return k

    k = max_contiguous_k(LIMIT)
    S = set(range(1, k + 1))

    # Candidates are numbers from k+1 to LIMIT
    candidates = list(range(k + 1, LIMIT + 1))

    # Sort by ascending largest prime factor, then by value
    candidates.sort(key=lambda x: (lpf[x], x))

    # Greedy pass 1: sorted by LPF
    for x in candidates:
        if time.time() > DEADLINE:
            break
        if can_add(x, S):
            S.add(x)

    # Greedy pass 2: try remaining candidates in numerical order
    # (in case ordering affected which could be added, though adding only constrains further)
    # Actually, since adding elements only makes it harder, one pass per ordering is sufficient.
    # But different orderings might yield different sets. Let's try another ordering.
    # However, we already have S from pass 1. We can't "undo" additions.
    # So we stick with the result of pass 1.

    # Optional: try a second greedy from scratch with a different ordering to see if we get more?
    # Given time constraints, we'll stick with the single pass but ensure it's efficient.
    # The LPF ordering is a strong heuristic for this type of problem.

    result = sorted(S)
    return result
# EVOLVE-BLOCK-END
