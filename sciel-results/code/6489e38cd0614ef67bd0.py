"""Two Sidon sets with disjoint nonzero difference sets.
Uses incremental SidonState for O(1) add/remove checks and
randomized local search with a time budget.
"""
import time
import random

N = 256
DEADLINE = 55.0  # safety margin before 60s kill


class SidonState:
    """Tracks pair sums and nonzero differences using bitmasks."""

    def __init__(self):
        self.els = 0       # bitmask of elements
        self.sums = 0      # bitmask of sums (indices 0..512)
        self.diffs = 0     # bitmask of diffs (indices 0..511, offset by 256)
        self.count = 0

    def can_add(self, x, other_diffs=None):
        if (self.els >> x) & 1:
            return False
        # Check sums
        new_sums = (1 << (2 * x)) | (self.els << x)
        if new_sums & self.sums:
            return False
        
        # Check diffs
        temp = self.els
        while temp:
            a = (temp & -temp).bit_length() - 1
            d1 = x - a
            d2 = a - x
            if (self.diffs >> (d1 + 256)) & 1:
                return False
            if (self.diffs >> (d2 + 256)) & 1:
                return False
            if other_diffs is not None:
                if (other_diffs >> (d1 + 256)) & 1:
                    return False
                if (other_diffs >> (d2 + 256)) & 1:
                    return False
            temp &= temp - 1
            
        return True

    def add(self, x):
        self.els |= (1 << x)
        self.sums |= (1 << (2 * x))
        self.count += 1
        temp = self.els & ~(1 << x)
        while temp:
            a = (temp & -temp).bit_length() - 1
            self.sums |= (1 << (a + x))
            self.diffs |= (1 << (x - a + 256))
            self.diffs |= (1 << (a - x + 256))
            temp &= temp - 1

    def remove(self, x):
        self.els &= ~(1 << x)
        self.sums &= ~(1 << (2 * x))
        self.count -= 1
        temp = self.els
        while temp:
            a = (temp & -temp).bit_length() - 1
            self.sums &= ~(1 << (a + x))
            self.diffs &= ~(1 << (x - a + 256))
            self.diffs &= ~(1 << (a - x + 256))
            temp &= temp - 1

    def size(self):
        return self.count

    def get_els_set(self):
        s = set()
        temp = self.els
        while temp:
            a = (temp & -temp).bit_length() - 1
            s.add(a)
            temp &= temp - 1
        return s


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
    """Try to grow A and B by adding/removing elements, with swaps."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add to A
        for x in range(1, N + 1):
            if ((A_state.els >> x) & 1) == 0 and A_state.size() < 64:
                if A_state.can_add(x, B_state.diffs):
                    A_state.add(x)
                    improved = True
                    break
        # Try to add to B
        if not improved:
            for x in range(1, N + 1):
                if ((B_state.els >> x) & 1) == 0 and B_state.size() < 64:
                    if B_state.can_add(x, A_state.diffs):
                        B_state.add(x)
                        improved = True
                        break
        # Try swap: remove a from A, add different x to A
        if not improved and time.time() < deadline:
            temp = A_state.els
            while temp:
                a = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                A_state.remove(a)
                added = False
                for x in range(1, N + 1):
                    if x != a and ((A_state.els >> x) & 1) == 0 and A_state.size() < 64:
                        if A_state.can_add(x, B_state.diffs):
                            A_state.add(x)
                            added = True
                            improved = True
                            break
                if not added:
                    A_state.add(a)
                if improved:
                    break
        # Try swap: remove b from B, add different x to B
        if not improved and time.time() < deadline:
            temp = B_state.els
            while temp:
                b = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                B_state.remove(b)
                added = False
                for x in range(1, N + 1):
                    if x != b and ((B_state.els >> x) & 1) == 0 and B_state.size() < 64:
                        if B_state.can_add(x, A_state.diffs):
                            B_state.add(x)
                            added = True
                            improved = True
                            break
                if not added:
                    B_state.add(b)
                if improved:
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
    for trial in range(2000):
        if time.time() > deadline - 2.0:
            break

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
            best_A = A_state.get_els_set()
            best_B = B_state.get_els_set()

        # Try reverse order: B first, then A
        if time.time() < deadline - 2.0:
            B_state2 = greedy_build(order_b, other_state=None)
            A_state2 = greedy_build(order_a, other_state=B_state2)
            A_state2, B_state2 = local_search(A_state2, B_state2, deadline - 1.0, rng)
            total2 = A_state2.size() + B_state2.size()
            if total2 > best_total:
                best_total = total2
                best_A = A_state2.get_els_set()
                best_B = B_state2.get_els_set()

    if best_A is None:
        # Fallback: simple greedy
        A_state = greedy_build(list(range(1, N + 1)))
        B_state = greedy_build(list(range(1, N + 1)), other_state=A_state)
        best_A = A_state.get_els_set()
        best_B = B_state.get_els_set()

    # Ensure nonempty
    if not best_A:
        best_A = {1}
    if not best_B:
        best_B = {2}

    return [sorted(best_A), sorted(best_B)]
