"""Reciprocal-subsum-free set task: maximize |S| for S subset of [1,24]
such that for no u in S does 1/u equal the sum of reciprocals of a
nonempty subset of {1/w : w in S, w != u}."""
import time
import random
from fractions import Fraction
from math import gcd

_START = None
DEADLINE = 78.0  # safety margin before 90s kill


def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


# LCM of 1..24 = 5354228880
_LCM = 5354228880
# Precompute integer reciprocal: 1/k as an integer numerator over _LCM
_REC = [_LCM // k for k in range(25)]


def _subset_sum_int(target_int, vals):
    """Exact subset-sum: can target_int be formed by a nonempty subset of
    positive ints vals? vals sorted ascending, target_int > 0."""
    n = len(vals)
    if n == 0:
        return False
    if target_int <= 0:
        return False
    # total bound
    if sum(vals) < target_int:
        return False
    if n <= 12:
        # DFS with pruning (vals ascending, prune when partial > target)
        best = [False]

        def dfs(i, s):
            if best[0]:
                return
            if s == target_int:
                best[0] = True
                return
            if s > target_int:
                return
            if i == n:
                return
            # take
            dfs(i + 1, s + vals[i])
            # skip
            dfs(i + 1, s)

        dfs(0, 0)
        return best[0]
    else:
        # Meet-in-the-middle
        half = n // 2
        left = vals[:half]
        right = vals[half:]

        left_sums = set()
        for mask in range(1 << len(left)):
            s = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
            left_sums.add(s)

        for mask in range(1 << len(right)):
            s = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += right[i]
            if (target_int - s) in left_sums:
                return True
        return False


def _is_valid_addition(S, v):
    """Check if adding v (the new maximum, all u in S < v) to sorted list S
    keeps the set reciprocal-subsum-free. Only new violations involve v as a
    summand for some u in S: 1/u = 1/v + sum(1/w for w in S, w > u)."""
    if not S:
        return True
    # pool of elements greater than u, for each u
    for u in S:
        target_int = _REC[u] - _REC[v]  # integer numerator of (1/u - 1/v)
        if target_int <= 0:
            continue
        pool = [w for w in S if w > u]
        if not pool:
            continue
        # quick bound
        if sum(_REC[w] for w in pool) < target_int:
            continue
        if _subset_sum_int(target_int, [_REC[w] for w in pool]):
            return False
    return True


def _is_valid_set(S):
    """Full validation of a set S (list of ints)."""
    if len(S) < 2:
        return True
    S_sorted = sorted(S)
    for i, u in enumerate(S_sorted):
        target_int = _REC[u]
        candidates = [w for j, w in enumerate(S_sorted) if j != i]
        if not candidates:
            continue
        if sum(_REC[w] for w in candidates) < target_int:
            continue
        if _subset_sum_int(target_int, [_REC[w] for w in candidates]):
            return False
    return True


def _greedy(order):
    """Greedy: add elements in given order, keeping validity."""
    S = []
    for v in order:
        if _deadline_hit():
            break
        if _is_valid_addition(S, v):
            S.append(v)
    return S


def _greedy_construct():
    """Multiple greedy orderings, keep the best."""
    best = []
    orders = [
        list(range(1, 25)),
        list(range(24, 0, -1)),
        list(range(2, 25)),
        list(range(24, 1, -1)),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
         3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 1],
        list(range(12, 0, -1)) + list(range(13, 25)),
        list(range(13, 25)) + list(range(12, 0, -1)),
    ]
    for order in orders:
        if _deadline_hit():
            break
        S = _greedy(order)
        if len(S) > len(best):
            best = S[:]
    return best


def _local_search(initial_set):
    """Local search: try to add elements, then try swaps to grow."""
    S = set(initial_set)
    best_set = list(S)
    best_size = len(S)

    improved = True
    while improved and not _deadline_hit():
        improved = False

        # Try adding one element
        for add in range(1, 25):
            if _deadline_hit():
                break
            if add in S:
                continue
            new_S = list(S | {add})
            if _is_valid_set(new_S):
                S = S | {add}
                best_set = list(S)
                best_size = len(S)
                improved = True

        # Try removing one and adding another (to grow)
        for remove in list(S):
            if _deadline_hit():
                break
            for add in range(1, 25):
                if add in S:
                    continue
                new_S = (S - {remove}) | {add}
                if len(new_S) <= best_size:
                    continue
                if _is_valid_set(list(new_S)):
                    S = new_S
                    best_set = list(S)
                    best_size = len(S)
                    improved = True
                    break
            if improved:
                break

    return best_set


def _random_greedy_trials(n_trials=30):
    """Random-order greedy, keep best."""
    best = []
    for _ in range(n_trials):
        if _deadline_hit():
            break
        order = list(range(1, 25))
        random.shuffle(order)
        S = _greedy(order)
        if len(S) > len(best):
            best = S[:]
    return best


def _branch_and_bound():
    """Branch and bound with incremental validity, descending order.
    Try to include each element; prune when remaining can't beat best."""
    candidates = list(range(24, 0, -1))
    best_set = []
    best_size = 0

    def backtrack(idx, S):
        nonlocal best_set, best_size
        if _deadline_hit():
            return
        # Prune: even taking all remaining can't beat best
        if len(S) + (len(candidates) - idx) <= best_size:
            return
        if idx == len(candidates):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return

        v = candidates[idx]
        # Option 1: include v (it's the new max of S since descending)
        if _is_valid_addition(S, v):
            backtrack(idx + 1, S + [v])
        # Option 2: exclude v
        backtrack(idx + 1, S)

    backtrack(0, [])
    return best_set


def construct():
    global _START
    _START = time.time()

    # Start with greedy
    result = _greedy_construct()

    # Improve with local search
    result = _local_search(result)

    # Random greedy trials
    if not _deadline_hit():
        rg = _random_greedy_trials(30)
        if len(rg) > len(result):
            result = _local_search(rg)

    # Branch and bound if time permits
    if not _deadline_hit():
        bb = _branch_and_bound()
        if len(bb) > len(result):
            result = bb

    # Final local search polish
    if not _deadline_hit():
        result = _local_search(result)

    # Safety: ensure nonempty
    if not result:
        result = [24]

    return result
