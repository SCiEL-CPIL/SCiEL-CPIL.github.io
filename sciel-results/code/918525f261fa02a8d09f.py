"""Two Sidon sets with disjoint nonzero difference sets.
Uses incremental SidonState for O(1) add/remove checks and
randomized local search with a time budget.
"""
import time
import random

N = 256
DEADLINE = 82.0  # safety margin before 90s kill


class SidonState:
    """Tracks pair sums (a+b, a<=b) and nonzero differences for a set."""

    def __init__(self):
        self.els = set()
        self.sums = set()
        self.diffs = set()

    def can_add(self, x, other_diffs=None):
        if x in self.els:
            return False
        # check sums: x+x and x+a for a in els
        if x + x in self.sums:
            return False
        for a in self.els:
            if a + x in self.sums:
                return False
        # check diffs: x-a and a-x for a in els
        for a in self.els:
            d = x - a
            if d in self.diffs:
                return False
            if other_diffs is not None and d in other_diffs:
                return False
        return True

    def add(self, x):
        self.els.add(x)
        self.sums.add(x + x)
        for a in list(self.els):
            if a != x:
                self.sums.add(a + x)
                self.diffs.add(x - a)
                self.diffs.add(a - x)

    def remove(self, x):
        self.els.discard(x)
        self.sums.discard(x + x)
        for a in list(self.els):
            self.sums.discard(a + x)
            self.diffs.discard(x - a)
            self.diffs.discard(a - x)

    def size(self):
        return len(self.els)


def greedy_build(order, other_state=None, max_size=64):
    """Greedy build in given order, respecting other_state's diffs."""
    state = SidonState()
    for x in order:
        if state.size() >= max_size:
            break
        if state.can_add(x, other_state.diffs if other_state else None):
            state.add(x)
    return state


def local_search(A_state, B_state, deadline, rng):
    """Try to grow A and B by adding/removing elements."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add to A
        for x in range(1, N + 1):
            if x not in A_state.els and A_state.size() < 64:
                if A_state.can_add(x, B_state.diffs):
                    A_state.add(x)
                    improved = True
                    break
        # Try to add to B
        for x in range(1, N + 1):
            if x not in B_state.els and B_state.size() < 64:
                if B_state.can_add(x, A_state.diffs):
                    B_state.add(x)
                    improved = True
                    break
    return A_state, B_state


def swap_improve(A_state, B_state, deadline, rng):
    """Try removing an element from one set to free diffs, then add to other."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try: remove from A, add to B
        for a in list(A_state.els):
            A_state.remove(a)
            # try adding to B
            added = False
            for x in range(1, N + 1):
                if x not in B_state.els and B_state.size() < 64:
                    if B_state.can_add(x, A_state.diffs):
                        B_state.add(x)
                        added = True
                        break
            # try adding back to A or other elements
            for x in range(1, N + 1):
                if x not in A_state.els and A_state.size() < 64:
                    if A_state.can_add(x, B_state.diffs):
                        A_state.add(x)
                        break
            # if we ended up with same or worse, revert
            if A_state.size() + B_state.size() < 2 * len(A_state.els) + B_state.size():
                pass  # complex revert; skip for now
            improved = True
            break
    return A_state, B_state


def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(42)

    best_A = None
    best_B = None
    best_total = 0

    # Multiple random restarts with different orderings
    for trial in range(200):
        if time.time() > deadline - 2.0:
            break

        # Random perturbation of ordering
        if trial == 0:
            order_a = list(range(1, N + 1))
            order_b = list(range(1, N + 1))
        else:
            order_a = list(range(1, N + 1))
            order_b = list(range(1, N + 1))
            rng.shuffle(order_a)
            rng.shuffle(order_b)

        # Build A first, then B with A's diffs as constraint
        A_state = greedy_build(order_a, other_state=None)
        B_state = greedy_build(order_b, other_state=A_state)

        # Local search to fill gaps
        A_state, B_state = local_search(A_state, B_state, deadline - 1.0, rng)

        total = A_state.size() + B_state.size()
        if total > best_total:
            best_total = total
            best_A = set(A_state.els)
            best_B = set(B_state.els)

        # Try reverse order: B first, then A
        if time.time() < deadline - 2.0:
            B_state2 = greedy_build(order_b, other_state=None)
            A_state2 = greedy_build(order_a, other_state=B_state2)
            A_state2, B_state2 = local_search(A_state2, B_state2, deadline - 1.0, rng)
            total2 = A_state2.size() + B_state2.size()
            if total2 > best_total:
                best_total = total2
                best_A = set(A_state2.els)
                best_B = set(B_state2.els)

    if best_A is None:
        # Fallback: simple greedy
        A_state = greedy_build(list(range(1, N + 1)))
        B_state = greedy_build(list(range(1, N + 1)), other_state=A_state)
        best_A = set(A_state.els)
        best_B = set(B_state.els)

    # Ensure nonempty
    if not best_A:
        best_A = {1}
    if not best_B:
        best_B = {2}

    # Cap at 64
    if len(best_A) > 64:
        best_A = set(list(best_A)[:64])
    if len(best_B) > 64:
        best_B = set(list(best_B)[:64])

    return [sorted(best_A), sorted(best_B)]
