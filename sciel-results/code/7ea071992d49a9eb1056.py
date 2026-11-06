"""Combined program: many randomized greedy restarts (proven mechanism) + fast swap local search."""
import time
import random
import numpy as np

N = 512
MAX = N

# Precompute for each x the list of (d) values for each of the 4 positions,
# so the inner loop is just iterating a precomputed tuple.
# pos_last[x]   : d such that x-3d >= 1  -> d in 1..(x-1)//3
# pos_first[x]  : d such that x+3d <= N  -> d in 1..(N-x)//3
# pos_second[x] : d such that x-d>=1 and x+2d<=N -> d in 1..min(x-1, (N-x)//2)
# pos_third[x]  : d such that x-2d>=1 and x+d<=N  -> d in 1..min((x-1)//2, N-x)

_pos_last   = [tuple(range(1, (x - 1) // 3 + 1)) for x in range(N + 1)]
_pos_first  = [tuple(range(1, (N - x) // 3 + 1)) for x in range(N + 1)]
_pos_second = [tuple(range(1, min(x - 1, (N - x) // 2) + 1)) for x in range(N + 1)]
_pos_third  = [tuple(range(1, min((x - 1) // 2, N - x) + 1)) for x in range(N + 1)]


def _is_4ap_free_np(selected):
    """Vectorized check using NumPy."""
    if not selected:
        return True
    arr = np.zeros(MAX + 1, dtype=bool)
    arr[list(selected)] = True
    s = np.nonzero(arr)[0]
    if len(s) == 0:
        return True
    # For each pair (a, b) with a<b in s, check if b-a = d and a+2d, a+3d in s
    # Vectorize over all a: for each a, d ranges over (s - a) where d>0
    # More efficient: for each a in s, for each d where a+3d<=N, check the 3 points.
    # Use broadcasting over a and d.
    a_arr = s.astype(np.int64)
    # d max for a: (N - a)//3
    d_max = (N - a_arr) // 3
    # Build all (a, d) pairs
    # Use a loop over a but vectorize over d for speed
    for a in a_arr:
        dm = (N - a) // 3
        if dm < 1:
            continue
        d = np.arange(1, dm + 1)
        t1 = a + d
        t2 = a + 2 * d
        t3 = a + 3 * d
        if np.any(arr[t1] & arr[t2] & arr[t3]):
            return False
    return True


def _greedy_forward(selected=None):
    """Fast forward greedy: only 'x is last term' can be violated in increasing order."""
    if selected is None:
        selected = set()
    for x in range(1, N + 1):
        if x in selected:
            continue
        ok = True
        for d in _pos_last[x]:
            if (x - d) in selected and (x - 2 * d) in selected and (x - 3 * d) in selected:
                ok = False
                break
        if ok:
            selected.add(x)
    return selected


def _greedy_reverse():
    """Reverse greedy: process from N down to 1, full 4-position check."""
    selected = set()
    for x in range(N, 0, -1):
        ok = True
        for d in _pos_first[x]:
            if (x + d) in selected and (x + 2 * d) in selected and (x + 3 * d) in selected:
                ok = False
                break
        if ok:
            for d in _pos_second[x]:
                if (x - d) in selected and (x + d) in selected and (x + 2 * d) in selected:
                    ok = False
                    break
        if ok:
            for d in _pos_third[x]:
                if (x - 2 * d) in selected and (x - d) in selected and (x + d) in selected:
                    ok = False
                    break
        if ok:
            for d in _pos_last[x]:
                if (x - d) in selected and (x - 2 * d) in selected and (x - 3 * d) in selected:
                    ok = False
                    break
        if ok:
            selected.add(x)
    return selected


def _greedy_random_order(rng):
    """Random-order greedy with full 4-position check."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    selected = set()
    for x in order:
        ok = True
        for d in _pos_last[x]:
            if (x - d) in selected and (x - 2 * d) in selected and (x - 3 * d) in selected:
                ok = False
                break
        if ok:
            for d in _pos_first[x]:
                if (x + d) in selected and (x + 2 * d) in selected and (x + 3 * d) in selected:
                    ok = False
                    break
        if ok:
            for d in _pos_second[x]:
                if (x - d) in selected and (x + d) in selected and (x + 2 * d) in selected:
                    ok = False
                    break
        if ok:
            for d in _pos_third[x]:
                if (x - 2 * d) in selected and (x - d) in selected and (x + d) in selected:
                    ok = False
                    break
        if ok:
            selected.add(x)
    return selected


def _can_add(x, selected):
    """Return True if x can be added to selected without forming a 4-AP."""
    for d in _pos_last[x]:
        if (x - d) in selected and (x - 2 * d) in selected and (x - 3 * d) in selected:
            return False
    for d in _pos_first[x]:
        if (x + d) in selected and (x + 2 * d) in selected and (x + 3 * d) in selected:
            return False
    for d in _pos_second[x]:
        if (x - d) in selected and (x + d) in selected and (x + 2 * d) in selected:
            return False
    for d in _pos_third[x]:
        if (x - 2 * d) in selected and (x - d) in selected and (x + d) in selected:
            return False
    return True


def _blockers_of(x, selected):
    """Return the set of elements in selected that, together with x, form a 4-AP."""
    blockers = set()
    for d in _pos_last[x]:
        a, b, c = x - d, x - 2 * d, x - 3 * d
        if a in selected and b in selected and c in selected:
            blockers.update((a, b, c))
    for d in _pos_first[x]:
        a, b, c = x + d, x + 2 * d, x + 3 * d
        if a in selected and b in selected and c in selected:
            blockers.update((a, b, c))
    for d in _pos_second[x]:
        a, b, c = x - d, x + d, x + 2 * d
        if a in selected and b in selected and c in selected:
            blockers.update((a, b, c))
    for d in _pos_third[x]:
        a, b, c = x - 2 * d, x - d, x + d
        if a in selected and b in selected and c in selected:
            blockers.update((a, b, c))
    return blockers


def _local_improve(selected, deadline):
    """Swap-based local search: try to add excluded elements by removing a single blocker,
    then greedily re-fill. Accept only if net size increases."""
    selected = set(selected)
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        # First pass: add any element that is not blocked
        for x in range(1, N + 1):
            if x not in selected and _can_add(x, selected):
                selected.add(x)
                improved = True
        # Second pass: try single-blocker swaps
        for x in range(1, N + 1):
            if time.time() > deadline - 3:
                break
            if x in selected:
                continue
            blockers = _blockers_of(x, selected)
            if not blockers:
                selected.add(x)
                improved = True
                continue
            # Try removing each single blocker b: remove b, add x, then greedily re-fill
            for b in blockers:
                if time.time() > deadline - 3:
                    break
                candidate = selected - {b}
                if not _can_add(x, candidate):
                    continue
                candidate.add(x)
                # Greedy re-fill
                changed = True
                while changed and time.time() < deadline - 3:
                    changed = False
                    for y in range(1, N + 1):
                        if y not in candidate and _can_add(y, candidate):
                            candidate.add(y)
                            changed = True
                if len(candidate) > len(selected):
                    selected = candidate
                    improved = True
                    break
    return selected


def construct():
    deadline = time.time() + 85  # Safety margin within 90s limit

    # 1) Deterministic forward greedy (fast)
    best_set = _greedy_forward()
    best_size = len(best_set)

    # 2) Reverse greedy
    rev = _greedy_reverse()
    if len(rev) > best_size:
        best_set, best_size = rev, len(rev)

    # 3) Many randomized restarts (the proven mechanism)
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 10:
        s = _greedy_random_order(rng)
        if len(s) > best_size:
            best_set, best_size = s, len(s)
        attempts += 1

    # 4) Local improvement on the best set found
    if time.time() < deadline - 5:
        improved = _local_improve(best_set, deadline)
        if len(improved) > best_size:
            best_set, best_size = improved, len(improved)

    # 5) Final verification; fallback if invalid
    if not _is_4ap_free_np(best_set):
        best_set = _greedy_forward()

    return sorted(best_set)
