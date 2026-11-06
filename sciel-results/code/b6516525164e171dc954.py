"""Minimum maximal no-isosceles set - structural rewrite using conflict-graph + local search."""

import time
import random
import numpy as np
import itertools

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
V = len(POINTS)  # 256

# Precompute squared distances
DIST2 = np.zeros((V, V), dtype=np.int64)
for i in range(V):
    xi, yi = POINTS[i]
    for j in range(i + 1, V):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        DIST2[i, j] = d
        DIST2[j, i] = d

# Precompute for each point p, a dictionary mapping squared distance to list of points
# This helps quickly check isosceles conditions
dist_groups = []
for p in range(V):
    dm = {}
    for q in range(V):
        if p == q:
            continue
        d = DIST2[p, q]
        if d not in dm:
            dm[d] = []
        dm[d].append(q)
    dist_groups.append(dm)


def _can_add(p, S_list):
    """Check if point p can be added to the set S_list (given as list of indices) without creating isosceles."""
    n = len(S_list)
    if n == 0:
        return True
    if n == 1:
        return True

    # Condition 1: all distances from p to S must be distinct
    d_seen = set()
    for a in S_list:
        d = DIST2[p, a]
        if d in d_seen:
            return False
        d_seen.add(d)

    # Condition 2: for each pair (a,b) in S, dist2(p,a) != dist2(a,b) and dist2(p,b) != dist2(a,b)
    for i in range(n):
        a = S_list[i]
        d_pa = DIST2[p, a]
        for j in range(i + 1, n):
            b = S_list[j]
            d_ab = DIST2[a, b]
            if d_pa == d_ab:
                return False
            d_pb = DIST2[p, b]
            if d_pb == d_ab:
                return False

    return True


def _is_blocked(u, S_list):
    """Check if point u is blocked by S_list (cannot be added without creating isosceles)."""
    n = len(S_list)
    if n < 2:
        return False

    d_ua = {}
    for a in S_list:
        d_ua[a] = DIST2[u, a]

    for i in range(n):
        a = S_list[i]
        d_ua_val = d_ua[a]
        for j in range(i + 1, n):
            b = S_list[j]
            d_ub_val = d_ua[b]
            d_ab = DIST2[a, b]
            if d_ua_val == d_ub_val or d_ua_val == d_ab or d_ub_val == d_ab:
                return True
    return False


def _is_valid(S_list):
    """Check if S_list is a valid no-isosceles set."""
    n = len(S_list)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                d_ij = DIST2[S_list[i], S_list[j]]
                d_ik = DIST2[S_list[i], S_list[k]]
                d_jk = DIST2[S_list[j], S_list[k]]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    return False
    return True


def _is_maximal(S_list):
    """Check if S_list is maximal (no point outside S can be added)."""
    S_set = set(S_list)
    for u in range(V):
        if u in S_set:
            continue
        if _can_add(u, S_list):
            return False
    return True


def _greedy_construct(order):
    """Build a no-isosceles set by processing points in given order, adding if possible."""
    S = []
    S_set = set()
    for p in order:
        if p in S_set:
            continue
        if _can_add(p, S):
            S.append(p)
            S_set.add(p)
    return S


def _reduce_set(S_list, rng, deadline):
    """Try to remove redundant points from S_list while maintaining maximality."""
    S = S_list[:]
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(S)):
            if time.time() > deadline - 5:
                break
            p = S[i]
            new_S = S[:i] + S[i + 1:]
            if _is_blocked(p, new_S):
                S = new_S
                changed = True
                break
    return S


def _swap_search(S_list, rng, deadline):
    """Try swapping one point out and another in, maintaining validity and maximality."""
    S = S_list[:]
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            if len(S) <= 1:
                break
            # Pick a point to remove
            idx = rng.randint(0, len(S) - 1)
            p_remove = S[idx]
            new_S = S[:idx] + S[idx + 1:]
            new_S_set = set(new_S)

            # Try to find a point to add that maintains maximality
            candidates = [p for p in range(V) if p not in new_S_set]
            rng.shuffle(candidates)

            found = False
            for p_add in candidates:
                if _can_add(p_add, new_S):
                    trial = new_S + [p_add]
                    # Check validity of the new set
                    if _is_valid(trial):
                        S = trial
                        improved = True
                        found = True
                        break
            if not found:
                # Keep the smaller set if it's still valid
                if _is_valid(new_S):
                    S = new_S
                    improved = True
    return S


def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill

    best_size = V
    best_set = []

    # Phase 1: Multiple greedy constructions with different orderings
    base_order = list(range(V))

    for restart in range(200):
        if time.time() > deadline - 30:
            break

        if restart % 5 == 0:
            # Different deterministic orderings
            if restart % 20 == 0:
                order = list(range(V))
            elif restart % 20 == 1:
                order = list(range(V - 1, -1, -1))
            elif restart % 20 == 2:
                # Row-major but reversed rows
                order = []
                for y in range(N - 1, -1, -1):
                    for x in range(N):
                        order.append(y * N + x)
            elif restart % 20 == 3:
                # Column-major
                order = []
                for x in range(N):
                    for y in range(N):
                        order.append(y * N + x)
            elif restart % 20 == 4:
                # Diagonal order
                order = []
                for d in range(2 * N - 1):
                    for x in range(max(0, d - N + 1), min(N, d + 1)):
                        y = d - x
                        order.append(y * N + x)
            else:
                order = base_order[:]
        else:
            # Random ordering
            order = base_order[:]
            random.Random(restart * 7 + 3).shuffle(order)

        S = _greedy_construct(order)

        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            if best_size <= 8:
                break

    # Phase 2: Reduce and swap-search on best set and nearby sets
    if best_set and time.time() < deadline - 15:
        # Reduce the best set
        reduced = _reduce_set(best_set, random.Random(42), deadline)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]

        # Swap search
        swapped = _swap_search(best_set, random.Random(123), deadline)
        if len(swapped) < best_size:
            best_size = len(swapped)
            best_set = swapped[:]

        # More restarts with reduction
        for restart in range(100):
            if time.time() > deadline - 10:
                break

            order = list(range(V))
            random.Random(10000 + restart).shuffle(order)
            S = _greedy_construct(order)
            if len(S) >= best_size:
                continue

            S = _reduce_set(S, random.Random(20000 + restart), deadline)
            if len(S) < best_size:
                best_size = len(S)
                best_set = S[:]

                # Try swap on this improved set
                S2 = _swap_search(S, random.Random(30000 + restart), deadline)
                if len(S2) < best_size:
                    best_size = len(S2)
                    best_set = S2[:]

            if best_size <= 8:
                break

    # Final verification
    if best_set:
        if _is_valid(best_set) and _is_maximal(best_set):
            return [POINTS[i] for i in best_set]

    # Fallback
    order = list(range(V))
    S = _greedy_construct(order)
    S = _reduce_set(S, random.Random(0), deadline)
    if _is_valid(S) and _is_maximal(S):
        return [POINTS[i] for i in S]

    # Ultimate fallback: return a simple valid maximal set
    return [POINTS[0]]
