"""Rewritten solver: incremental validity + many random-restart saturation passes."""
import time
import random
from itertools import combinations

N = 192
MAX_SIZE = 48
MAX_TRIPLE_SUM = 190 + 191 + 192  # 573


class Solver:
    __slots__ = ("set", "triple_set", "pair_res")

    def __init__(self):
        self.set = set()
        self.triple_set = set()   # all sums of 3 distinct elements currently in the set
        self.pair_res = {}        # a -> set of (i+j) % a for pairs i,j in set, i!=j, i!=a, j!=a

    def can_add(self, x):
        # Condition for a = x: no 3-subset of current set sums to a multiple of x.
        for m in range(x, MAX_TRIPLE_SUM + 1, x):
            if m in self.triple_set:
                return False
        # Condition for each existing a: no pair (in set\{a}) plus x sums to a multiple of a,
        # i.e. no pair with (i+j) % a == (-x) % a.
        for a in self.set:
            if ((-x) % a) in self.pair_res[a]:
                return False
        return True

    def add(self, x):
        old = list(self.set)
        # new triple sums involving x
        for i, j in combinations(old, 2):
            self.triple_set.add(x + i + j)
        # update pair residues for existing elements
        for a in self.set:
            pr = self.pair_res[a]
            for i in old:
                if i != a:
                    pr.add((x + i) % a)
        # pair residues for x
        prx = set()
        for i, j in combinations(old, 2):
            prx.add((i + j) % x)
        self.pair_res[x] = prx
        self.set.add(x)


def build(seed, strategy, deadline):
    rng = random.Random(seed)
    sol = Solver()

    # initial greedy ordering
    if strategy == 0:
        order = list(range(N, 0, -1))
    elif strategy == 1:
        order = list(range(1, N + 1))
    elif strategy == 2:
        order = list(range(1, N + 1))
        rng.shuffle(order)
    else:
        # prefer numbers with fewer divisors (primes / hard divisors) first, then fill
        order = list(range(1, N + 1))
        order.sort(key=lambda v: (sum(1 for d in range(2, v + 1) if v % d == 0), v), reverse=True)

    for x in order:
        if time.time() > deadline:
            break
        if len(sol.set) >= MAX_SIZE:
            break
        if sol.can_add(x):
            sol.add(x)

    # saturation passes with random orderings
    passes = 0
    while time.time() < deadline and len(sol.set) < MAX_SIZE and passes < 12:
        passes += 1
        unused = [x for x in range(1, N + 1) if x not in sol.set]
        if not unused:
            break
        rng.shuffle(unused)
        added = False
        for x in unused:
            if time.time() > deadline:
                break
            if len(sol.set) >= MAX_SIZE:
                break
            if sol.can_add(x):
                sol.add(x)
                added = True
        if not added:
            break

    return sorted(sol.set)


def _simple_greedy():
    selected = []
    for x in range(N, 0, -1):
        candidate = selected + [x]
        ok = True
        for a in candidate:
            others = [v for v in candidate if v != a]
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.append(x)
    return selected


def construct():
    deadline = time.time() + 78  # safety margin under the 90s kill
    best = None
    best_len = -1

    strategies = [0, 2, 1, 3]
    seed = 0
    while time.time() < deadline - 3:
        strat = strategies[seed % len(strategies)]
        cand = build(seed, strat, deadline)
        if len(cand) > best_len:
            best_len = len(cand)
            best = cand
        if best_len >= MAX_SIZE:
            break
        seed += 1

    if best is None or best_len < 1:
        best = _simple_greedy()

    return best
