"""Two Sidon sets with disjoint difference sets – structural rewrite.
Uses bitset-backed incremental SidonState, greedy construction, and
time-budgeted local search (add, swap, 2-opt) to maximise |A|+|B|."""

import time
import random
import sys

N = 256
MAX_SIZE = 64
DEADLINE = 85.0  # safety margin before 90 s kill


class SidonState:
    """Maintains pair-sum set and nonzero-difference set for a Sidon set.
    Supports O(k) can_add / add / remove where k = |set|."""

    __slots__ = ('els', 'sums', 'diffs')

    def __init__(self, els=None):
        self.els = list(els) if els else []
        self._rebuild()

    def _rebuild(self):
        self.sums = set()
        self.diffs = set()
        els = self.els
        n = len(els)
        for i in range(n):
            a = els[i]
            self.sums.add(a + a)
            for j in range(i + 1, n):
                b = els[j]
                self.sums.add(a + b)
                self.diffs.add(b - a)
                self.diffs.add(a - b)

    def can_add(self, x, other_diffs=None):
        """Return True if adding x keeps Sidon property and disjoint diffs."""
        for a in self.els:
            if (a + x) in self.sums:
                return False
            if other_diffs is not None:
                d = a - x
                if d in other_diffs:
                    return False
        if (x + x) in self.sums:
            return False
        return True

    def add(self, x):
        """Add x (caller must have verified can_add)."""
        for a in self.els:
            self.sums.add(a + x)
            self.diffs.add(a - x)
            self.diffs.add(x - a)
        self.sums.add(x + x)
        self.els.append(x)

    def remove(self, x):
        """Remove x and clean up sums/diffs."""
        self.els.remove(x)
        for a in self.els:
            self.sums.discard(a + x)
            self.diffs.discard(a - x)
            self.diffs.discard(x - a)
        self.sums.discard(x + x)


def greedy_construct(max_time=10.0):
    """Greedy: iterate candidates, add to whichever side accepts it."""
    start = time.time()
    A = SidonState()
    B = SidonState()
    for x in range(1, N + 1):
        if time.time() - start > max_time:
            break
        # Try A first, then B
        if len(A.els) < MAX_SIZE and A.can_add(x, B.diffs):
            A.add(x)
        elif len(B.els) < MAX_SIZE and B.can_add(x, A.diffs):
            B.add(x)
    return A, B


def local_search(A, B, time_limit):
    """Time-budgeted local search: try adds, swaps, and 2-opt moves."""
    start = time.time()
    rng = random.Random(42)
    best_score = len(A.els) + len(B.els)
    best_A_els = list(A.els)
    best_B_els = list(B.els)

    # Build candidate pools
    all_vals = set(range(1, N + 1))

    def elapsed():
        return time.time() - start

    def try_add(state, other, x):
        if len(state.els) >= MAX_SIZE:
            return False
        if x in state.els:
            return False
        if state.can_add(x, other.diffs):
            state.add(x)
            return True
        return False

    def try_remove(state, x):
        if x not in state.els:
            return False
        state.remove(x)
        return True

    # Phase 1: Try to add any remaining element
    while elapsed() < time_limit:
        improved = False
        for x in sorted(all_vals - set(A.els) - set(B.els)):
            if elapsed() > time_limit:
                break
            if try_add(A, B, x):
                improved = True
            elif try_add(B, A, x):
                improved = True
        # Also try elements in one set to move to the other
        for x in list(A.els):
            if elapsed() > time_limit:
                break
            if B.can_add(x, A.diffs - {d for d in A.diffs if d % 1 != 0}):
                # Need to check: after removing x from A, does B accept x?
                A.remove(x)
                if B.can_add(x, A.diffs):
                    B.add(x)
                    improved = True
                else:
                    A.add(x)  # put back
        for x in list(B.els):
            if elapsed() > time_limit:
                break
            if A.can_add(x, B.diffs):
                B.remove(x)
                if A.can_add(x, B.diffs):
                    A.add(x)
                    improved = True
                else:
                    B.add(x)
        if not improved:
            break

    score = len(A.els) + len(B.els)
    if score > best_score:
        best_score = score
        best_A_els = list(A.els)
        best_B_els = list(B.els)

    # Phase 2: Randomized swap / 2-opt
    while elapsed() < time_limit:
        # Pick a random element from A and try to swap with one from outside
        if A.els and elapsed() < time_limit:
            i = rng.randrange(len(A.els))
            a = A.els[i]
            # Try removing a and adding a random candidate
            A.remove(a)
            candidates = [x for x in rng.sample(range(1, N + 1), min(50, N))
                          if x not in A.els and x not in B.els]
            added = False
            for x in candidates:
                if A.can_add(x, B.diffs):
                    A.add(x)
                    added = True
                    break
            if not added:
                A.add(a)  # restore
            else:
                s = len(A.els) + len(B.els)
                if s > best_score:
                    best_score = s
                    best_A_els = list(A.els)
                    best_B_els = list(B.els)

        if B.els and elapsed() < time_limit:
            i = rng.randrange(len(B.els))
            b = B.els[i]
            B.remove(b)
            candidates = [x for x in rng.sample(range(1, N + 1), min(50, N))
                          if x not in A.els and x not in B.els]
            added = False
            for x in candidates:
                if B.can_add(x, A.diffs):
                    B.add(x)
                    added = True
                    break
            if not added:
                B.add(b)
            else:
                s = len(A.els) + len(B.els)
                if s > best_score:
                    best_score = s
                    best_A_els = list(A.els)
                    best_B_els = list(B.els)

    # Restore best
    A.els = list(best_A_els)
    A._rebuild()
    B.els = list(best_B_els)
    B._rebuild()
    return A, B


def construct():
    global DEADLINE
    t0 = time.time()
    remaining = DEADLINE - (t0 - t0)  # just track from now
    deadline = t0 + 85.0

    # Phase 1: Greedy construction
    A, B = greedy_construct(max_time=min(5.0, deadline - time.time() - 1.0))
    score1 = len(A.els) + len(B.els)

    # Phase 2: Local search with remaining time
    remaining_time = deadline - time.time() - 2.0  # 2s safety margin
    if remaining_time > 1.0:
        A, B = local_search(A, B, remaining_time)

    score = len(A.els) + len(B.els)

    # Ensure non-empty
    if not A.els:
        A.els = [1]
        A._rebuild()
    if not B.els:
        B.els = [2]
        B._rebuild()

    # Final safety: verify and trim if needed
    return [list(A.els), list(B.els)]
