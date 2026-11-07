"""Rewritten program: primes + maximal composite set with no two-other-product divisibility."""
# EVOLVE-BLOCK-START
import time
import math

_N = 256
_TIME_LIMIT = 80.0  # wall-clock safety margin (evaluator kills at 90s)


def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i * i:n + 1:i] = bytearray(len(sieve[i * i:n + 1:i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def _validity_ok(setlist):
    """Check the full contract for a list of ints (distinct, in 2..N, no a|b*c for distinct a,b,c)."""
    if not setlist:
        return False
    s = set(setlist)
    if len(s) != len(setlist):
        return False
    for x in s:
        if not (2 <= x <= _N):
            return False
    L = sorted(s)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                if (L[j] * L[k]) % a == 0:
                    return False
    return True


def _can_add(a, members):
    """Return True if adding a (distinct from all members) keeps the set valid.
    members is a list of ints (all < a in greedy usage)."""
    # a must not divide b*c for any distinct b,c in members (b != c)
    n = len(members)
    for j in range(n):
        b = members[j]
        for k in range(n):
            if k == j:
                continue
            if (b * members[k]) % a == 0:
                return False
    return True


def _greedy(order, base, deadline):
    """Greedily add candidates in `order` (list of ints) on top of `base` (list of ints)."""
    members = list(base)
    for a in order:
        if time.time() > deadline:
            break
        # a must not already be present
        if a in members:
            continue
        if _can_add(a, members):
            members.append(a)
    return members


def construct():
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT

    primes = _primes_upto(_N)
    composites = [x for x in range(2, _N + 1) if x not in set(primes)]

    best = None
    best_len = -1

    # Candidate orderings to try (all are deterministic and fast)
    orderings = []
    # 1) composites descending (prefer large composites to preserve small-number capacity)
    orderings.append(list(reversed(composites)))
    # 2) composites ascending
    orderings.append(list(composites))
    # 3) composites by (number of divisors ascending, value descending)
    def _divcount(x):
        c = 0
        for d in range(1, int(math.isqrt(x)) + 1):
            if x % d == 0:
                c += 1
                if d * d != x:
                    c += 1
        return c
    orderings.append(sorted(composites, key=lambda x: (_divcount(x), -x)))
    # 4) composites by (value descending, divisors ascending)
    orderings.append(sorted(composites, key=lambda x: (-x, _divcount(x))))

    for order in orderings:
        if time.time() > deadline:
            break
        cand = _greedy(order, primes, deadline)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
        # early stop if we hit a perfect bound (all primes + all composites = 255)
        if best_len == _N - 1:
            break

    if best is None:
        best = primes

    # Final exact validity check; fall back to primes if anything is off.
    if not _validity_ok(best):
        best = primes

    # Return as a Python list of plain ints.
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
