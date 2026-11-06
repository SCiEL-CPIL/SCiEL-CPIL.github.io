import time
import random
import heapq

def construct():
    N = 1000
    start = time.time()
    DEADLINE = start + 142.0   # safety margin before 150s kill

    # ------------------------------------------------------------------
    # Sparse representation.
    #   S        : set of chosen elements
    #   reps     : dict sum -> list of (a,b) pairs with a<=b, a,b in S
    #   For feasibility we only ever need to know whether a sum is "full"
    #   (len==2).  We keep the full lists so we can do removals exactly.
    # ------------------------------------------------------------------

    class B2:
        __slots__ = ("S", "reps", "full")

        def __init__(self):
            self.S = set()
            self.reps = {}
            self.full = set()      # sums currently at exactly 2 reps

        def _add_pair(self, a, b):
            s = a + b
            lst = self.reps.get(s)
            if lst is None:
                self.reps[s] = [(a, b)]
            else:
                lst.append((a, b))
                if len(lst) == 2:
                    self.full.add(s)

        def _remove_pair(self, a, b):
            s = a + b
            lst = self.reps[s]
            lst.remove((a, b))
            if len(lst) == 1:
                self.full.discard(s)
            if not lst:
                del self.reps[s]

        def can_add(self, x):
            # adding x creates pair (x,x) and (x,y) for y in S, y!=x
            if x + x in self.full:
                return False
            for y in self.S:
                if y != x and (x + y) in self.full:
                    return False
            return True

        def add(self, x):
            self.S.add(x)
            self._add_pair(x, x)
            for y in self.S:
                if y != x:
                    self._add_pair(x, y)

        def remove(self, x):
            for y in self.S:
                if y != x:
                    self._remove_pair(x, y)
            self._remove_pair(x, x)
            self.S.discard(x)

    # ------------------------------------------------------------------
    # Greedy maximal construction.
    #   Order candidates by a "spread" heuristic: prefer elements that are
    #   far apart in value (larger gaps -> fewer collisions in sums).
    #   We iterate over a shuffled-but-structure-aware order and add any
    #   element that is feasible, giving a maximal (not just greedy) set.
    # ------------------------------------------------------------------

    def greedy_maximal(order, seed):
        b = B2()
        rng = random.Random(seed)
        # We want a maximal independent set: scan in `order`, add if feasible.
        for x in order:
            if b.can_add(x):
                b.add(x)
        return b

    def make_orders():
        """Produce several candidate orderings for the greedy pass."""
        base = list(range(1, N + 1))
        orders = []

        # 1) pure random
        r = random.Random(7)
        o = base[:]; r.shuffle(o); orders.append(o)

        # 2) random with a mild preference for larger gaps:
        #    sort by a random key but bias toward spread
        r = random.Random(11)
        o = sorted(base, key=lambda v: (r.random(), v))
        orders.append(o)

        # 3) stride-based orders (deterministic spread)
        for step in (2, 3, 4, 5):
            o = list(range(1, N + 1, step)) + list(range(1, N + 1, step))[::2]
            # de-dup while preserving order
            seen = set(); ded = []
            for v in o:
                if v not in seen:
                    seen.add(v); ded.append(v)
            orders.append(ded)

        # 4) a few more random shuffles
        for s in (23, 29, 31):
            r = random.Random(s)
            o = base[:]; r.shuffle(o); orders.append(o)

        return orders

    # ------------------------------------------------------------------
    # Local improvement: 2-opt style.
    #   Find an element x in S whose removal unblocks the most candidates,
    #   remove it, then greedily re-add the best candidates (possibly
    #   including x again and new ones) to try to grow the set.
    # ------------------------------------------------------------------

    def unblock_score(b, x):
        """How many currently-infeasible candidates become feasible if x removed.
        Approximate cheaply: count candidates y not in S for which x is the
        sole blocker, i.e. some sum involving x is full and that sum also
        involves y.  We approximate by: for each full sum s that involves x,
        count how many y (not in S) could use s.  This is O(degree)."""
        # sums involving x: 2x and x+y for y in S, y!=x
        involved = {2 * x}
        involved.update(x + y for y in b.S if y != x)
        score = 0
        for s in involved:
            if s in b.full:
                # the two reps of s are (a,b) pairs; removing x frees one slot
                # candidates that could fill it: any y not in S with y + z = s
                # for some z in S\{x} or y=y.  Cheap upper bound: number of
                # distinct partners in S (minus x) plus possibly y=y/2.
                partners = sum(1 for (a, c) in b.reps[s] for _ in [0])  # 2 reps
                # each rep (a,c) gives candidate c (if c!=x) and a (if a!=x)
                cands = set()
                for (a, c) in b.reps[s]:
                    for v in (a, c):
                        if v != x and v not in b.S:
                            cands.add(v)
                score += len(cands)
        return score

    def improve(b, budget, seed):
        rng = random.Random(seed)
        S = list(b.S)
        best = len(b.S)
        # repeatedly try to find a beneficial 2-opt move
        while time.time() < budget:
            if not S:
                break
            # score each element by how much it blocks; pick top few
            scored = [(unblock_score(b, x), x) for x in S]
            scored.sort(reverse=True)
            progressed = False
            for sc, x in scored[:8]:
                if time.time() > budget:
                    break
                if sc <= 0:
                    continue
                # remove x, then greedily re-add best candidates
                b.remove(x)
                # candidate pool: all not-in-S, sorted by a spread heuristic
                cand = [v for v in range(1, N + 1) if v not in b.S]
                # try to add as many as possible, preferring x back and spread
                # simple: iterate candidates in random order, add if feasible,
                # but cap the work
                added = []
                rng2 = random.Random(seed + x)
                rng2.shuffle(cand)
                for v in cand:
                    if b.can_add(v):
                        b.add(v)
                        added.append(v)
                if len(b.S) > best:
                    best = len(b.S)
                    S = list(b.S)
                    progressed = True
                    break
                # if no growth, undo: rebuild is expensive; instead we keep
                # the change only if it helps.  Roll back by recomputing.
                # Easiest: revert to previous S.
                # We'll just undo by removing what we added and re-adding x.
                for v in added:
                    b.remove(v)
                b.add(x)
            if not progressed:
                # try a few random removals to escape
                for _ in range(3):
                    if time.time() > budget:
                        break
                    x = rng.choice(S)
                    b.remove(x)
                    cand = [v for v in range(1, N + 1) if v not in b.S]
                    rng.shuffle(cand)
                    added = []
                    for v in cand:
                        if b.can_add(v):
                            b.add(v)
                            added.append(v)
                    if len(b.S) > best:
                        best = len(b.S)
                        S = list(b.S)
                        break
                    else:
                        for v in added:
                            b.remove(v)
                        b.add(x)
        return best

    # ------------------------------------------------------------------
    # Driver: run several greedy seeds, keep the best, then improve.
    # ------------------------------------------------------------------
    orders = make_orders()
    best_size = 0
    best_b = None
    per_seed = 8.0

    for i, order in enumerate(orders):
        if time.time() > DEADLINE - 30.0:
            break
        b = greedy_maximal(order, seed=100 + i)
        s = len(b.S)
        if s > best_size:
            best_size = s
            best_b = b

    # Improve the best construction, with a shrinking time budget.
    if best_b is not None:
        remaining = DEADLINE - time.time()
        if remaining > 5.0:
            # spend most of the remaining budget on improvement
            improve(best_b, time.time() + max(0.0, remaining - 4.0), seed=999)
            best_size = len(best_b.S)

    # Final: return sorted list of the best set.
    if best_b is None:
        return list(range(1, N + 1, 2))  # trivial valid fallback (all odd)
    return sorted(best_b.S)


if __name__ == "__main__":
    print(len(construct()))
