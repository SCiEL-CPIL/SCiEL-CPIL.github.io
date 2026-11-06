"""Two Sidon sets with disjoint difference sets.
Incremental state tracking + randomized greedy restarts + local-search polish.
"""
import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 85.0  # safety margin before 90s kill
START = time.time()


class SidonState:
    __slots__ = ("els", "sums", "diffs")

    def __init__(self):
        self.els = []
        self.sums = set()   # all a+b for a<=b
        self.diffs = set()  # all nonzero a-b

    def can_add(self, x, other_diffs):
        for a in self.els:
            s = a + x
            if s in self.sums:
                return False
            d = a - x
            if d in self.diffs or d in other_diffs:
                return False
        if (x + x) in self.sums:
            return False
        return True

    def add(self, x, other_diffs):
        for a in self.els:
            self.sums.add(a + x)
            d = a - x
            self.diffs.add(d)
            self.diffs.add(-d)
        self.sums.add(x + x)
        self.els.append(x)

    def remove_last(self):
        x = self.els.pop()
        self.sums.discard(x + x)
        for a in self.els:
            self.sums.discard(a + x)
            self.diffs.discard(a - x)
            self.diffs.discard(x - a)


def _time_left():
    return DEADLINE - (time.time() - START)


def _greedy_build(order_a, order_b, other_order):
    """Build A and B greedily. other_order = the order of the OTHER set's
    elements (used to precompute diffs as they get added)."""
    sa = SidonState()
    sb = SidonState()
    for x in order_a:
        if len(sa.els) >= MAX_SIZE:
            break
        if sa.can_add(x, sb.diffs):
            sa.add(x, sb.diffs)
    for x in order_b:
        if len(sb.els) >= MAX_SIZE:
            break
        if sb.can_add(x, sa.diffs):
            sb.add(x, sa.diffs)
    return sa, sb


def _local_search(sa, sb, budget):
    """Try to add elements that were skipped, or swap to free up room."""
    used = set(sa.els) | set(sb.els)
    # try to add any unused element to either set
    for x in range(1, N + 1):
        if x in used:
            continue
        if len(sa.els) < MAX_SIZE and sa.can_add(x, sb.diffs):
            sa.add(x, sb.diffs)
            used.add(x)
        elif len(sb.els) < MAX_SIZE and sb.can_add(x, sa.diffs):
            sb.add(x, sa.diffs)
            used.add(x)
    return sa, sb


def construct():
    best_a, best_b = None, None
    best_total = 0
    rng = random.Random(12345)

    while _time_left() > 2.0:
        # Random permutation of candidates
        order_a = list(range(1, N + 1))
        order_b = list(range(1, N + 1))
        rng.shuffle(order_a)
        rng.shuffle(order_b)

        sa, sb = _greedy_build(order_a, order_b, None)
        sa, sb = _local_search(sa, sb, None)

        total = len(sa.els) + len(sb.els)
        if total > best_total:
            best_total = total
            best_a, best_b = list(sa.els), list(sb.els)

        # Early stop if we hit the theoretical max
        if best_total >= 2 * MAX_SIZE:
            break

    if best_a is None:
        # Fallback: trivial valid sets
        best_a = [1]
        best_b = [2]

    return [best_a, best_b]
