import time
import random
from fractions import Fraction

DEADLINE = 85.0
_START = None

def _start_clock():
    global _START
    _START = time.time()

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _subset_sum_possible(vals, target):
    """True iff some subset of `vals` (Fractions) sums exactly to `target`."""
    n = len(vals)
    if n == 0:
        return target == Fraction(0)
    mid = n // 2
    left, right = vals[:mid], vals[mid:]
    
    L = {Fraction(0)}
    for v in left:
        L = L | {x + v for x in L}
    R = {Fraction(0)}
    for v in right:
        R = R | {x + v for x in R}
    
    if len(L) <= len(R):
        for l in L:
            if (target - l) in R:
                return True
    else:
        for r in R:
            if (target - r) in L:
                return True
    return False


def is_valid(S):
    """Exact reciprocal-subsum-free check for a nonempty set S of ints."""
    S = list(set(S))
    if not S:
        return False
    r = {v: Fraction(1, v) for v in S}
    for a in S:
        others = [r[v] for v in S if v != a]
        if _subset_sum_possible(others, r[a]):
            return False
    return True


def _quick_necessary_check(S):
    """Quick necessary condition: for each a, 1/a must be <= sum of all other reciprocals.
    If 1/a > sum of all others, it can't be a subset sum (since all reciprocals are positive).
    Actually, we need 1/a to be representable, so if 1/a > sum of all others, it's impossible.
    But we also need to check that it's not too small... actually all subset sums are positive,
    so if 1/a is less than the minimum reciprocal, it can't be formed. But the minimum is 1/max(S).
    This is a weak check. A better one: sort and check prefix sums."""
    S = list(set(S))
    if not S:
        return False
    S.sort()
    r = [Fraction(1, v) for v in S]
    # For each element, check if its reciprocal could possibly be a subset sum
    # of the others. A necessary condition: 1/a must be <= sum of all other reciprocals
    for i, a in enumerate(S):
        others_sum = sum(r[j] for j in range(len(S)) if j != i)
        if r[i] > others_sum:
            # 1/a is larger than the sum of all others, so it can't be a subset sum
            # This is actually a SUFFICIENT condition for validity of this element!
            pass
        # Another check: if 1/a is smaller than the smallest other reciprocal,
        # it can't be a subset sum (since all terms are positive and >= 1/max)
        min_other = min(r[j] for j in range(len(S)) if j != i) if len(S) > 1 else None
        if min_other is not None and r[i] < min_other:
            # 1/a < 1/max_others, so 1/a < min subset sum of any nonempty subset
            # This means 1/a can't be formed
            pass
    return True  # Always passes; this is just a placeholder for potential optimization


def greedy_add(order, current_best_size):
    """Grow a set by adding elements in `order`, keeping validity."""
    chosen = []
    recips = []
    for v in order:
        if v in chosen:
            continue
        # Quick check: 1/v must not be a subset sum of current reciprocals
        if recips and _subset_sum_possible(recips, Fraction(1, v)):
            continue
        trial = chosen + [v]
        if len(trial) <= current_best_size:
            continue
        if is_valid(trial):
            chosen = trial
            recips.append(Fraction(1, v))
    return chosen


def greedy_remove():
    """Start from [1,24]; remove elements until valid."""
    cur = set(range(1, 25))
    while not is_valid(cur) and not _deadline_hit():
        changed = False
        # Try removing each element, see if it helps
        for v in sorted(cur, reverse=True):
            trial = cur - {v}
            if is_valid(trial):
                cur = trial
                changed = True
                break
        if not changed:
            break
    return list(cur)


def local_search(S, max_iter=100):
    """Try to improve set S by adding, removing, or swapping elements."""
    S = set(S)
    best = S
    for _ in range(max_iter):
        if _deadline_hit():
            break
        improved = False
        
        # Try adding elements
        for v in range(1, 25):
            if v not in S:
                trial = S | {v}
                if is_valid(trial) and len(trial) > len(best):
                    best = trial
                    S = trial
                    improved = True
                    break
        
        # Try swapping: remove one, add one
        if not improved:
            for remove_v in list(S):
                if _deadline_hit():
                    break
                for add_v in range(1, 25):
                    if add_v in S or add_v == remove_v:
                        continue
                    trial = (S - {remove_v}) | {add_v}
                    if is_valid(trial) and len(trial) > len(best):
                        best = trial
                        S = trial
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best


def construct():
    _start_clock()
    
    rng = random.Random(42)
    best_set = set()
    
    def consider(c):
        nonlocal best_set
        c = set(c)
        if not c:
            return
        if len(c) <= len(best_set):
            return
        if is_valid(c):
            best_set = c
    
    # 1) Start with top-half anchor
    consider(list(range(13, 25)))
    
    # 2) Greedy add from multiple orders
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        list(range(24, 0, -2)) + list(range(23, 0, -2)),
        [24, 12, 23, 11, 22, 10, 21, 9, 20, 8, 19, 7, 18, 6, 17, 5, 16, 4, 15, 3, 14, 2, 13, 1],
    ]
    for order in orders:
        if _deadline_hit():
            break
        consider(greedy_add(order, len(best_set)))
    
    # 3) Random shuffle greedy adds
    for _ in range(100):
        if _deadline_hit():
            break
        order = list(range(1, 25))
        rng.shuffle(order)
        consider(greedy_add(order, len(best_set)))
    
    # 4) Greedy removal from full set
    if not _deadline_hit():
        consider(greedy_remove())
    
    # 5) Local search refinement on best set
    if best_set and not _deadline_hit():
        improved = local_search(list(best_set), max_iter=50)
        if len(improved) > len(best_set):
            best_set = improved
    
    # 6) One more pass: try to add any remaining element
    if not _deadline_hit():
        remaining = [v for v in range(1, 25) if v not in best_set]
        remaining.sort(reverse=True)
        for v in remaining:
            if _deadline_hit():
                break
            trial = best_set | {v}
            if is_valid(trial):
                best_set = trial
    
    # Ensure nonempty
    if not best_set:
        best_set = set(range(13, 25))
    
    return list(best_set)


if __name__ == "__main__":
    print(construct())
