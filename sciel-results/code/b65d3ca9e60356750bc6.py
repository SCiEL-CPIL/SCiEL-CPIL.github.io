"""Optimized solver for equal-lcm-triple avoidance on [1,128]."""
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    t0 = time.time()
    DEADLINE = t0 + 82.0  # safety margin before 90s kill

    N = 128
    # Precompute lcm table for all pairs (1-indexed, 0 unused)
    lcm_tab = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            v = a * b // gcd(a, b)
            lcm_tab[a][b] = v
            lcm_tab[b][a] = v

    def has_bad_triple(S):
        """Check if set S contains a bad triple."""
        L = len(S)
        if L < 3:
            return False
        for i in range(L):
            a = S[i]
            for j in range(i + 1, L):
                b = S[j]
                ab = lcm_tab[a][b]
                for k in range(j + 1, L):
                    c = S[k]
                    if lcm_tab[a][c] == ab and lcm_tab[b][c] == ab:
                        return True
        return False

    def can_add(S, x, S_set):
        """Check if adding x to S creates a bad triple."""
        L = len(S)
        for i in range(L):
            a = S[i]
            xa = lcm_tab[x][a]
            for j in range(i + 1, L):
                b = S[j]
                if lcm_tab[x][b] == xa and lcm_tab[a][b] == xa:
                    return False
        return True

    def greedy(order):
        S = []
        S_set = set()
        for x in order:
            if can_add(S, x, S_set):
                S.append(x)
                S_set.add(x)
        return S

    def local_search(S):
        """Try to add any missing element; then try swaps to grow."""
        S = list(S)
        S_set = set(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S_set and can_add(S, x, S_set):
                    S.append(x)
                    S_set.add(x)
                    changed = True
        # Try removing one element and adding more
        improved = True
        while improved:
            improved = False
            for i in range(len(S)):
                a = S[i]
                S2 = S[:i] + S[i+1:]
                S2_set = set(S2)
                # Try to add all possible
                for x in range(1, N + 1):
                    if x not in S2_set and can_add(S2, x, S2_set):
                        S2.append(x)
                        S2_set.add(x)
                if len(S2) > len(S):
                    S = S2
                    S_set = S2_set
                    improved = True
                    break
        return S

    def try_swap(S):
        """Try removing two elements and adding three (or more)."""
        S = list(S)
        S_set = set(S)
        improved = True
        while improved:
            improved = False
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    a, b = S[i], S[j]
                    S2 = S[:i] + S[i+1:j] + S[j+1:]
                    S2_set = set(S2)
                    for x in range(1, N + 1):
                        if x not in S2_set and can_add(S2, x, S2_set):
                            S2.append(x)
                            S2_set.add(x)
                    if len(S2) > len(S):
                        S = S2
                        S_set = S2_set
                        improved = True
                        break
                if improved:
                    break
        return S

    # Build candidate orderings
    def is_prime(x):
        if x < 2:
            return False
        d = 2
        while d * d <= x:
            if x % d == 0:
                return False
            d += 1
        return True

    primes = [x for x in range(2, N + 1) if is_prime(x)]
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    evens = [x for x in range(2, N + 1, 2)]

    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        primes + [x for x in range(1, N + 1) if not is_prime(x)],
        odds + evens,
        evens + odds,
        sorted(range(1, N + 1), key=lambda x: (x.bit_count(), x)),
        sorted(range(1, N + 1), key=lambda x: -x.bit_count()),
        # Prime powers first
        [p**k for p in primes for k in range(1, 8) if p**k <= N] +
        [x for x in range(1, N + 1) if all(x != p**k for p in primes for k in range(1, 8) if p**k <= N)],
    ]

    best = []
    for order in orderings:
        if time.time() > DEADLINE - 5:
            break
        S = greedy(order)
        if len(S) > len(best):
            best = S

    # Local search polish
    if time.time() < DEADLINE - 5:
        best = local_search(best)

    if time.time() < DEADLINE - 3:
        best = try_swap(best)

    # Final add pass
    S_set = set(best)
    for x in range(1, N + 1):
        if x not in S_set and can_add(best, x, S_set):
            best.append(x)
            S_set.add(x)

    # Verify
    if has_bad_triple(best):
        # Fallback: primes + 1 + powers of 2
        fb = [1] + primes
        for k in range(2, 8):
            v = 1 << k
            if v <= N and v not in fb:
                fb.append(v)
        best = fb

    if not best:
        best = [1]

    return sorted(best)
