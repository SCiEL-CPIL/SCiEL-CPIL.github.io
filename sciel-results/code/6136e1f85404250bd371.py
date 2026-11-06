"""Structural redesign: greedy set-cover over 1..L using best (smallest) prime per residue,
with binary search for the largest L fully coverable within reciprocal budget 1."""
import time
import math
from fractions import Fraction

_T0 = time.time()
_DEADLINE = _T0 + 112.0          # explicit wall-clock guard (well under 150 s kill)
_CAP = 100000


def _sieve_primes(limit):
    flag = bytearray([1]) * (limit + 1)
    flag[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if flag[i]:
            step = i
            start = i * i
            flag[start::step] = b'\x00' * (((limit - start) // step) + 1)
    return [i for i in range(2, limit + 1) if flag[i]]


def _build_best(primes):
    """best[r] = smallest prime p (<=1000) with p % r == 0, i.e. covering n with n % p == r."""
    best = {}
    for p in primes:                       # ascending -> first (smallest) wins
        for r in range(p):
            if r not in best:
                best[r] = p
    return best


def _covered_len(used, L):
    """Longest prefix 1..L fully covered by residue classes in `used` (dict r->p)."""
    best = used
    cnt = 0
    for n in range(1, L + 1):
        if best.get(n) is not None:
            cnt += 1
        else:
            break
    return cnt


def construct():
    primes = _sieve_primes(1000)
    best = _build_best(primes)

    # ---- quick greedy at cap to get an upper bound and a seed solution ----
    def greedy(L):
        used = {}
        budget = Fraction(0)
        while True:
            cur = _covered_len(used, L)
            if cur >= L:
                break
            if time.time() > _DEADLINE:
                break
            nxt = cur + 1
            cand = best.get(nxt)
            if cand is None:
                break
            cost = Fraction(1, cand)
            if budget + cost > 1:
                break
            used[nxt] = cand
            budget += cost
        return used, budget, _covered_len(used, L)

    used, budget, cov = greedy(_CAP)
    best_used, best_cov = used, cov

    if time.time() > _DEADLINE:
        return [[p, r] for r, p in used.items()]

    # ---- binary search the largest L with a full cover ----
    lo, hi = 1, _CAP
    while lo <= hi and time.time() < _DEADLINE:
        mid = (lo + hi) // 2
        u, b, c = greedy(mid)
        if c >= mid:
            best_used, best_cov = u, c
            lo = mid + 1
        else:
            hi = mid - 1

    return [[p, r] for r, p in best_used.items()]


if __name__ == "__main__":
    res = construct()
    # self-check
    from collections import defaultdict
    seen = set()
    tot = Fraction(0)
    for p, r in res:
        assert p <= 1000 and 0 <= r < p and p not in seen, (p, r)
        seen.add(p)
        tot += Fraction(1, p)
    assert tot <= 1, tot
    covered = set()
    for p, r in res:
        for n in range(r if r >= 1 else p, 100001, p):
            covered.add(n)
    L = 0
    for n in range(1, 100001):
        if n in covered:
            L += 1
        else:
            break
    print("classes:", len(res), "budget:", float(tot), "L:", L)
