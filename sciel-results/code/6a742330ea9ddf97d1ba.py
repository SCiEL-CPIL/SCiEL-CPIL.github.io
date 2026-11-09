"""Covering 1..L by prime residue classes with reciprocal budget 1 (CSP / gap-driven backtracking)."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def _covered_prefix_len(classes, cap=100000):
    if not classes:
        return 0
    covered = bytearray(cap + 1)
    for p, r in classes:
        start = r if r >= 1 else p
        for x in range(start, cap + 1, p):
            covered[x] = 1
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def construct():
    t0 = time.time()
    DEADLINE = 143.0  # safety margin before 150s kill
    MAXL = 100000
    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}
    budget0 = Fraction(1, 1)

    # Fast incremental coverage tracker with count array (O(p) add/remove).
    class Tracker:
        __slots__ = ("maxl", "count", "L", "state")
        def __init__(self, maxl):
            self.maxl = maxl
            self.count = [0] * (maxl + 1)
            self.L = 0
            self.state = {}

        def _start(self, p, r):
            return r if r >= 1 else p

        def add(self, p, r):
            if p in self.state:
                return False
            start = self._start(p, r)
            for x in range(start, self.maxl + 1, p):
                self.count[x] += 1
            self.state[p] = r
            self._extend_L()
            return True

        def remove(self, p):
            if p not in self.state:
                return False
            r = self.state[p]
            start = self._start(p, r)
            for x in range(start, self.maxl + 1, p):
                self.count[x] -= 1
            del self.state[p]
            self._recompute_L()
            return True

        def change_residue(self, p, new_r):
            if p not in self.state:
                return False
            old_r = self.state[p]
            if old_r == new_r:
                return False
            for x in range(self._start(p, old_r), self.maxl + 1, p):
                self.count[x] -= 1
            for x in range(self._start(p, new_r), self.maxl + 1, p):
                self.count[x] += 1
            self.state[p] = new_r
            self._recompute_L()
            return True

        def _extend_L(self):
            newL = self.L
            for i in range(self.L + 1, self.maxl + 1):
                if self.count[i] > 0:
                    newL = i
                else:
                    break
            self.L = newL

        def _recompute_L(self):
            newL = 0
            for i in range(1, self.maxl + 1):
                if self.count[i] > 0:
                    newL = i
                else:
                    break
            self.L = newL

        def first_gap(self):
            for i in range(self.L + 1, self.maxl + 1):
                if self.count[i] == 0:
                    return i
            return None

        def snapshot(self):
            return dict(self.state)

        def restore(self, snap):
            self.state = dict(snap)
            for i in range(self.maxl + 1):
                self.count[i] = 0
            for p, r in self.state.items():
                for x in range(self._start(p, r), self.maxl + 1, p):
                    self.count[x] += 1
            self._recompute_L()

    # ---- Phase 1: Greedy initialization to get a strong starting frontier ----
    def greedy_init(seed):
        rng = random.Random(seed)
        tr = Tracker(MAXL)
        budget = budget0
        order = primes[:]
        rng.shuffle(order)
        for _ in range(len(primes)):
            if time.time() > DEADLINE:
                break
            best_gain = 0
            best_p = None
            best_r = None
            for p in order:
                if p in tr.state or budget < recip[p]:
                    continue
                target = tr.L + 1
                r = target % p
                K = 2000
                gain = 0
                for x in range(tr.L + 1, min(tr.L + K, MAXL) + 1):
                    if tr.count[x] == 0 and (x - r) % p == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            if best_p is None or best_gain == 0:
                break
            tr.add(best_p, best_r)
            budget -= recip[best_p]
        return tr, budget

    best_tr = None
    best_L = 0
    for seed in range(30):
        if time.time() > DEADLINE:
            break
        tr, budget = greedy_init(seed)
        if tr.L > best_L:
            best_L = tr.L
            best_tr = tr
            best_budget = budget

    if best_tr is None:
        primes13 = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes13)]

    tr = best_tr
    budget = best_budget
    best_L_ls = tr.L
    best_state = tr.snapshot()
    best_budget_ls = budget

    rng = random.Random(7)

    # ---- Phase 2: CSP-style gap-driven backtracking ----
    # Forward-checking helper: given a gap position g and a set of available
    # primes (unused + used), decide which prime is the cheapest way to cover g.
    # We branch on the choice of prime to cover the current first gap.
    def solve_gap(tr, budget, depth, max_depth):
        """Recursively cover the first gap. Returns (success, new_budget)."""
        if time.time() > DEADLINE:
            return False, budget
        gap = tr.first_gap()
        if gap is None:
            return True, budget
        if depth >= max_depth:
            return False, budget

        # Candidate primes: those whose residue can cover `gap` and fit budget.
        # For a used prime p, the residue is fixed; it already either covers gap
        # (then gap wouldn't be the first gap) or doesn't. So only unused primes
        # can newly cover gap by choosing r = gap % p.
        candidates = []
        for p in primes:
            if p in tr.state:
                continue
            if budget < recip[p]:
                continue
            candidates.append(p)

        if not candidates:
            return False, budget

        # Heuristic ordering: cheapest (largest p) first to preserve budget,
        # but among cheap ones prefer those that also cover the next gap.
        def score_cand(p):
            r = gap % p
            # count how many positions in [gap, gap+200] it newly covers
            c = 0
            for x in range(gap, min(gap + 200, MAXL) + 1):
                if tr.count[x] == 0 and (x - r) % p == 0:
                    c += 1
            return (-c, -p)  # more new coverage first, then larger p (cheaper)

        candidates.sort(key=score_cand)

        for p in candidates:
            r = gap % p
            old_L = tr.L
            tr.add(p, r)
            nb = budget - recip[p]
            if tr.L > best_L_ls:
                pass  # will be recorded by caller
            ok, nb2 = solve_gap(tr, nb, depth + 1, max_depth)
            if ok:
                return True, nb2
            tr.remove(p)
        return False, budget

    # ---- Phase 3: Iterative deepening + random perturbation to escape optima ----
    max_no_improve = 0
    no_improve = 0
    base_max_depth = 4
    while time.time() < DEADLINE:
        gap = tr.first_gap()
        if gap is None:
            break
        # Try CSP solve with increasing depth
        solved = False
        for md in range(base_max_depth, base_max_depth + 6):
            if time.time() > DEADLINE:
                break
            # snapshot to restore on failure
            snap = tr.snapshot()
            ok, nb = solve_gap(tr, budget, 0, md)
            if ok:
                solved = True
                if tr.L > best_L_ls:
                    best_L_ls = tr.L
                    best_state = tr.snapshot()
                    best_budget_ls = budget
                break
            else:
                tr.restore(snap)
                budget = best_budget_ls if False else budget  # budget unchanged on restore
                # recompute budget from state
                budget = budget0
                for p in tr.state:
                    budget -= recip[p]
        if tr.L > best_L_ls:
            best_L_ls = tr.L
            best_state = tr.snapshot()
            budget = budget0
            for p in tr.state:
                budget -= recip[p]
            best_budget_ls = budget
            no_improve = 0
        else:
            no_improve += 1

        # Escape: random perturbation of a residue
        if tr.state:
            p = rng.choice(list(tr.state.keys()))
            old_r = tr.state[p]
            new_r = rng.randrange(p)
            tr.change_residue(p, new_r)
            if tr.L < best_L_ls - 200:
                tr.change_residue(p, old_r)

        if no_improve > 200:
            # reset to best and deepen
            tr.restore(best_state)
            budget = best_budget_ls
            no_improve = 0
            base_max_depth = min(base_max_depth + 1, 10)

    # Final budget/consistency check
    result = [[p, r] for p, r in best_state.items()]
    s = Fraction(0, 1)
    for p, _ in result:
        s += Fraction(1, p)
    if s > budget0:
        # fallback to a valid small set
        primes13 = [p for p in primes if p >= 13]
        result = [[p, (i + 1) % p] for i, p in enumerate(primes13)]
    return result
# EVOLVE-BLOCK-END
