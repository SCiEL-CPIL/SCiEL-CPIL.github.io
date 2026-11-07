"""Large sets without a triple and all its pair sums.
Maximize |S|, S subset of 1..128, distinct, with no three distinct
members a,b,c such that a+b, a+c, b+c are all in S.
"""
import time
import random

_N = 128

# Precompute, for each sum s, the list of unordered pairs (a,b), a<b, a+b=s.
PAIRS = [[] for _ in range(_N + 1)]
for _s in range(2, _N + 1):
    for _a in range(1, _s // 2 + 1):
        _b = _s - _a
        if _b > _a and _b <= _N:
            PAIRS[_s].append((_a, _b))


def can_add(x, Sset):
    """Return True if adding x (not in Sset) keeps the set valid."""
    # Case 1: x is a member of the forbidden triple {x, a, b}.
    # Need distinct a,b in Sset with x+a, x+b, a+b all in Sset.
    A = [a for a in Sset if (x + a) in Sset]
    nA = len(A)
    for i in range(nA):
        ai = A[i]
        for j in range(i + 1, nA):
            if (ai + A[j]) in Sset:
                return False
    # Case 2: x is a pair-sum a+b of a triple {a,b,c} subset Sset.
    # Need a+b==x (a,b in Sset) and some c in Sset (c!=a,b) with
    # a+c in Sset and b+c in Sset.
    for (a, b) in PAIRS[x]:
        if a in Sset and b in Sset:
            for c in Sset:
                if c != a and c != b and (a + c) in Sset and (b + c) in Sset:
                    return False
    return True


def greedy(order, Sset0=None):
    Sset = set(Sset0) if Sset0 is not None else set()
    for x in order:
        if x in Sset:
            continue
        if can_add(x, Sset):
            Sset.add(x)
    return Sset


def greedy_adaptive():
    """Repeatedly add the least-constraining remaining number (monotonic)."""
    Sset = set()
    remaining = list(range(1, _N + 1))
    while remaining:
        best = None
        bestscore = None
        for x in remaining:
            score = 0
            for a in Sset:
                if (x + a) in Sset:
                    score += 1
            for (a, b) in PAIRS[x]:
                if a in Sset and b in Sset:
                    score += 1
            if (bestscore is None or score < bestscore
                    or (score == bestscore and (best is None or x > best))):
                bestscore = score
                best = x
        if can_add(best, Sset):
            Sset.add(best)
        remaining.remove(best)
    return Sset


def add_closure(Sset, order):
    """Add any number that fits, in `order`, until maximal."""
    changed = True
    passes = 0
    while changed and passes < 12:
        changed = False
        passes += 1
        for x in order:
            if x in Sset:
                continue
            if can_add(x, Sset):
                Sset.add(x)
                changed = True
    return Sset


def is_valid(Sset):
    L = sorted(Sset)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(i + 1, n):
            b = L[j]
            for k in range(j + 1, n):
                c = L[k]
                if (a + b) in Sset and (a + c) in Sset and (b + c) in Sset:
                    return False
    return True


def construct():
    t0 = time.time()
    DEADLINE = 70.0  # safety margin under the 90s kill
    best = None

    def consider(Sset):
        nonlocal best
        Sset = add_closure(Sset, list(range(1, _N + 1)))
        if best is None or len(Sset) > len(best):
            best = Sset

    # Deterministic strong starts.
    consider(greedy_adaptive())
    consider(greedy(list(range(_N, 0, -1))))
    consider(greedy(list(range(1, _N + 1))))

    # Random restarts to explore different maximal sets.
    rng = random.Random(20240517)
    while time.time() - t0 < DEADLINE:
        order = list(range(1, _N + 1))
        rng.shuffle(order)
        clo = list(range(1, _N + 1))
        rng.shuffle(clo)
        Sset = greedy(order)
        Sset = add_closure(Sset, clo)
        if best is None or len(Sset) > len(best):
            best = Sset

    if best is None:
        best = set(range(_N // 2 + 1, _N + 1))
    else:
        best = add_closure(best, list(range(1, _N + 1)))

    # Final safety: guarantee a valid result.
    if not is_valid(best):
        best = set(range(_N // 2 + 1, _N + 1))

    return [int(x) for x in sorted(best)]
