"""CSP backtracking search for A,B with max distinct pairwise-coprime sums.
Structural idea: rank-1 sum grid s_ij = a_i + b_j; enforce pairwise coprimality
incrementally via prime-factor-disjointness (a smooth surrogate for the min-gcd
objective: we require all shared prime factors to be empty)."""
import time
import math
import random

_DEADLINE = 82.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(s):
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            if _gcd(si, s[j]) != 1:
                return False
    return True

def _is_valid(A, B):
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    for x in A:
        if x < 1 or x > 128:
            return False
    for x in B:
        if x < 1 or x > 128:
            return False
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) < 2:
        return False
    return _pairwise_coprime(sorted(sums))

def _score(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    return len(sums)

def _factorize(n):
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

def _prime_factors(n):
    # precomputed below; placeholder
    return _PF[n]

# Precompute prime factor sets for all possible sums (2..256)
_PF = {}
def _build_pf():
    for n in range(2, 257):
        _PF[n] = _factorize(n)
_build_pf()

def _col_pf(a, B):
    """Prime-factor union of the three sums a+B[0], a+B[1], a+B[2].
    Also return the three sums and whether they are internally pairwise coprime."""
    s0 = a + B[0]
    s1 = a + B[1]
    s2 = a + B[2]
    if _gcd(s0, s1) != 1 or _gcd(s0, s2) != 1 or _gcd(s1, s2) != 1:
        return None
    pf = _PF[s0] | _PF[s1] | _PF[s2]
    return (a, (s0, s1, s2), pf)

def _solve():
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _score(best_A, best_B)  # known good: 5

    # ---- Phase 1: CSP backtracking over B for a set of structured A candidates ----
    # Build A candidates: small structured + random, each internally "spread"
    A_candidates = []
    for a1 in range(1, 30):
        for a2 in range(a1 + 1, 30):
            for a3 in range(a2 + 1, 30):
                A_candidates.append((a1, a2, a3))
    random.seed(42)
    for _ in range(800):
        A_candidates.append(tuple(sorted(random.sample(range(1, 129), 3))))
    random.shuffle(A_candidates)

    # For each A, do a backtracking search over B (b1<b2<b3) with incremental
    # prime-factor-disjointness across the three columns.
    for A in A_candidates:
        if _elapsed() > _DEADLINE:
            break
        a1, a2, a3 = A
        # Precompute, for each b in [1,128], the column's sums + prime-factor set
        # (relative to this A). A column is "valid" if its 3 sums are pairwise coprime.
        col = {}
        for b in range(1, 129):
            s0 = a1 + b
            s1 = a2 + b
            s2 = a3 + b
            if _gcd(s0, s1) != 1 or _gcd(s0, s2) != 1 or _gcd(s1, s2) != 1:
                continue
            pf = _PF[s0] | _PF[s1] | _PF[s2]
            col[b] = ((s0, s1, s2), pf)
        if len(col) < 3:
            continue
        cols = sorted(col.items())  # list of (b, (sums, pf))
        n = len(cols)
        # Backtrack: choose b1<b2<b3 with mutually disjoint pf sets.
        # Use incremental: pick b1, then b2 with disjoint pf, then b3.
        found = False
        for i in range(n):
            if found or _elapsed() > _DEADLINE:
                break
            b1, (sums1, pf1) = cols[i]
            for j in range(i + 1, n):
                if found or _elapsed() > _DEADLINE:
                    break
                b2, (sums2, pf2) = cols[j]
                if pf1 & pf2:
                    continue
                for k in range(j + 1, n):
                    b3, (sums3, pf3) = cols[k]
                    if pf1 & pf3 or pf2 & pf3:
                        continue
                    B = [b1, b2, b3]
                    sc = _score(A, B)
                    if sc > best_score:
                        best_score = sc
                        best_A = [a1, a2, a3]
                        best_B = B
                        if sc >= 9:
                            found = True
                            break
                    # We keep searching for a better B for this A (don't stop at first)
        if best_score >= 9:
            break

    if best_score >= 2 and _is_valid(best_A, best_B):
        return [best_A, best_B]

    # ---- Phase 2: symmetric CSP over A for structured B candidates ----
    B_candidates = []
    for b1 in range(1, 30):
        for b2 in range(b1 + 1, 30):
            for b3 in range(b2 + 1, 30):
                B_candidates.append((b1, b2, b3))
    random.seed(1234)
    for _ in range(800):
        B_candidates.append(tuple(sorted(random.sample(range(1, 129), 3))))
    random.shuffle(B_candidates)

    for B in B_candidates:
        if _elapsed() > _DEADLINE:
            break
        b1, b2, b3 = B
        col = {}
        for a in range(1, 129):
            s0 = a + b1
            s1 = a + b2
            s2 = a + b3
            if _gcd(s0, s1) != 1 or _gcd(s0, s2) != 1 or _gcd(s1, s2) != 1:
                continue
            pf = _PF[s0] | _PF[s1] | _PF[s2]
            col[a] = ((s0, s1, s2), pf)
        if len(col) < 3:
            continue
        cols = sorted(col.items())
        n = len(cols)
        for i in range(n):
            if _elapsed() > _DEADLINE:
                break
            a1, (sums1, pf1) = cols[i]
            for j in range(i + 1, n):
                if _elapsed() > _DEADLINE:
                    break
                a2, (sums2, pf2) = cols[j]
                if pf1 & pf2:
                    continue
                for k in range(j + 1, n):
                    a3, (sums3, pf3) = cols[k]
                    if pf1 & pf3 or pf2 & pf3:
                        continue
                    A = [a1, a2, a3]
                    sc = _score(A, B)
                    if sc > best_score:
                        best_score = sc
                        best_A = A
                        best_B = [b1, b2, b3]
                        if sc >= 9:
                            return [best_A, best_B]

    if best_score >= 2 and _is_valid(best_A, best_B):
        return [best_A, best_B]

    # ---- Phase 3: fallback local search / brute small ----
    for a1 in range(1, 20):
        for a2 in range(a1 + 1, 20):
            for a3 in range(a2 + 1, 20):
                for b1 in range(1, 20):
                    for b2 in range(b1 + 1, 20):
                        for b3 in range(b2 + 1, 20):
                            A = [a1, a2, a3]
                            B = [b1, b2, b3]
                            if _is_valid(A, B):
                                sc = _score(A, B)
                                if sc > best_score:
                                    best_A = A
                                    best_B = B
                                    best_score = sc
    if best_score >= 2:
        return [best_A, best_B]

    return [[1, 2, 3], [1, 2, 3]]

def construct():
    try:
        result = _solve()
        if _is_valid(result[0], result[1]):
            return result
    except Exception:
        pass
    return [[1, 2, 3], [1, 2, 3]]
