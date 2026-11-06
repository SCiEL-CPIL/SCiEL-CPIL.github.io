"""Two Sidon sets with disjoint difference sets.

Combines an efficient incremental SidonState (O(|S|) add check) with a
refinement strategy: interleaved per-candidate greedy plus local search
(single adds, single-element swaps, and 1-for-2 exchanges).
"""

import time
import random

DEADLINE = 85.0  # safety margin before the 90s kill
N = 256
MAX_SIZE = 64


class SidonState:
    """Tracks pair sums and nonzero differences for a set, with O(|S|) add."""

    def __init__(self):
        self.els = []
        self.sums = set()      # all a+b for a<=b
        self.diffs = set()     # all a-b for a!=b (nonzero)
        self.member = set()

    def _check_pair(self, a, x, other_diffs):
        """Return False if adding x next to existing a breaks constraints."""
        if (a + x) in self.sums:
            return False
        d1 = a - x
        d2 = x - a
        if d1 in self.diffs or d2 in self.diffs:
            return False
        if other_diffs is not None:
            if d1 in other_diffs or d2 in other_diffs:
                return False
        return True

    def can_add(self, x, other_diffs=None):
        """Check if x can be added maintaining Sidon property and disjoint diffs."""
        if x in self.member:
            return False
        for a in self.els:
            if not self._check_pair(a, x, other_diffs):
                return False
        if (x + x) in self.sums:
            return False
        return True

    def add(self, x):
        """Add x to the set. Assumes can_add returned True."""
        for a in self.els:
            self.sums.add(a + x)
            d1 = a - x
            d2 = x - a
            self.diffs.add(d1)
            self.diffs.add(d2)
        self.sums.add(x + x)
        self.els.append(x)
        self.member.add(x)
        self.els.sort()

    def remove(self, x):
        """Remove x from the set."""
        if x not in self.member:
            return
        self.member.discard(x)
        self.els.remove(x)
        to_remove_sums = set()
        to_remove_diffs = set()
        for a in self.els:
            to_remove_sums.add(a + x)
            d1 = a - x
            d2 = x - a
            to_remove_diffs.add(d1)
            to_remove_diffs.add(d2)
        to_remove_sums.add(x + x)
        self.sums -= to_remove_sums
        self.diffs -= to_remove_diffs
        self.els.sort()

    def size(self):
        return len(self.els)


def interleaved_greedy(deadline, rng, target_a=None, target_b=None):
    """Build A and B simultaneously, trying each candidate on both sides."""
    state_a = SidonState()
    state_b = SidonState()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)

    for x in candidates:
        if state_a.size() >= MAX_SIZE and state_b.size() >= MAX_SIZE:
            break
        if time.time() > deadline:
            break
        if state_a.size() < MAX_SIZE and state_a.can_add(x, state_b.diffs):
            state_a.add(x)
        if state_b.size() < MAX_SIZE and state_b.can_add(x, state_a.diffs):
            state_b.add(x)

    return state_a, state_b


def try_add(state, other, deadline):
    """Try to add any new element to `state` (respecting other's diffs)."""
    if state.size() >= MAX_SIZE:
        return False
    for x in range(1, N + 1):
        if time.time() > deadline:
            return False
        if x in state.member:
            continue
        if state.can_add(x, other.diffs):
            state.add(x)
            return True
    return False


def try_swap(state, other, deadline):
    """Remove one element, then add one (net zero) to free difference slots."""
    if not state.els:
        return False
    for x in state.els:
        if time.time() > deadline:
            return False
        state.remove(x)
        added = False
        for y in range(1, N + 1):
            if y in state.member:
                continue
            if state.can_add(y, other.diffs):
                state.add(y)
                added = True
                break
        if not added:
            state.add(x)  # restore
    return False


def try_exchange(state, other, deadline):
    """Remove one element, add two (net +1) to grow the smaller set."""
    if state.size() >= MAX_SIZE or not state.els:
        return False
    for x in state.els:
        if time.time() > deadline:
            return False
        state.remove(x)
        count_before = state.size()
        added = 0
        for y in range(1, N + 1):
            if y in state.member:
                continue
            if state.can_add(y, other.diffs):
                state.add(y)
                added += 1
                if state.size() >= MAX_SIZE:
                    break
        if added >= 1:
            return True  # net growth (or at least no loss)
        else:
            state.add(x)  # restore
    return False


def local_search(state_a, state_b, deadline, rng):
    """Improve combined size via adds, swaps, and 1-for-2 exchanges."""
    stall = 0
    while time.time() < deadline:
        progress = False
        # 1) plain adds
        if try_add(state_a, state_b, deadline):
            progress = True
        if try_add(state_b, state_a, deadline):
            progress = True
        # 2) grow the smaller set via exchange
        if state_a.size() <= state_b.size():
            if try_exchange(state_a, state_b, deadline):
                progress = True
        else:
            if try_exchange(state_b, state_a, deadline):
                progress = True
        # 3) swaps to free slots
        if try_swap(state_a, state_b, deadline):
            progress = True
        if try_swap(state_b, state_a, deadline):
            progress = True

        if not progress:
            stall += 1
            if stall > 2:
                break
        else:
            stall = 0
    return state_a, state_b


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)

    best_total = 0
    best_a = []
    best_b = []

    def record(state_a, state_b):
        nonlocal best_total, best_a, best_b
        total = state_a.size() + state_b.size()
        if total > best_total:
            best_total = total
            best_a = list(state_a.els)
            best_b = list(state_b.els)

    restart = 0
    while time.time() < deadline - 3.0:
        restart += 1
        state_a, state_b = interleaved_greedy(deadline - 3.0, rng)
        record(state_a, state_b)

        # Local search only while we have a comfortable time margin.
        if time.time() < deadline - 8.0:
            state_a, state_b = local_search(state_a, state_b, deadline - 8.0, rng)
            record(state_a, state_b)

    # Ensure nonempty
    if not best_a:
        best_a = [1]
    if not best_b:
        best_b = [2]

    return [best_a, best_b]
