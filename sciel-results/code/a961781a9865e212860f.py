import time
import random
from math import gcd


def _lcm(a, b):
    return a * b // gcd(a, b)


def construct():
    deadline = time.time() + 80.0
    N = 128
    nums = list(range(1, N + 1))
    rng = random.Random(20240611)

    # For each x, list unordered pairs (a,b) such that {a,b,x} is forbidden
    # (all three pairwise lcms equal).
    conflict_with = {x: [] for x in range(1, N + 1)}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = _lcm(a, b)
            for x in range(1, N + 1):
                if x == a or x == b:
                    continue
                if _lcm(a, x) == lab and _lcm(b, x) == lab:
                    conflict_with[x].append((a, b))

    def can_add(x, S):
        for a, b in conflict_with[x]:
            if a in S and b in S:
                return False
        return True

    def greedy(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def add_phase(S):
        S = set(S)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in nums:
                if x not in S and can_add(x, S):
                    S.add(x)
                    changed = True
        return S

    def one_opt(S):
        S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S = add_phase(S)
            for a in list(S):
                if time.time() > deadline:
                    break
                S2 = add_phase(S - {a})
                if len(S2) > len(S):
                    S = S2
                    improved = True
                    break
        return S

    # Prime powers are mutually safe -> strong base ordering.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
              109, 113, 127]
    pp = [1]
    for p in primes:
        v = p
        while v <= N:
            pp.append(v)
            v *= p
    pp_set = set(pp)
    order_pp = pp + [x for x in nums if x not in pp_set]

    orderings = [
        nums,
        sorted(nums, key=lambda x: len(conflict_with[x])),
        sorted(nums, key=lambda x: -len(conflict_with[x])),
        sorted(nums, reverse=True),
        order_pp,
    ]

    best = set()
    for o in orderings:
        S = add_phase(greedy(o))
        if len(S) > len(best):
            best = set(S)
    best = one_opt(best)

    # Random restarts to escape local optima.
    while time.time() < deadline:
        o = nums[:]
        rng.shuffle(o)
        S = add_phase(greedy(o))
        if len(S) > len(best):
            best = set(S)
            best = one_opt(best)

    if not best:
        best = {1}
    return sorted(best)
