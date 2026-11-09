"""Combined approach: NumPy-accelerated no-isosceles check + minimum-degree greedy
+ swap-based local search for maximal no-isosceles sets on [0,16)^2.

Key structural ideas:
- Vectorized distance checks (NumPy) for fast _can_add.
- Minimum-degree greedy: pick the addable point that blocks the fewest others.
- Swap local search: try replacing one set point with a non-set point if it keeps
  the set no-isosceles and still maximal, reducing size or improving structure.
- Remove-and-re-maximize local search to shrink the set.
- Multi-start with deadline guard.
"""

import time
import random
import itertools
import numpy as np

N = 16
DEADLINE_SECONDS = 290  # safety margin before 300s kill

# Precompute all points and indices
ALL_POINTS = np.array(list(itertools.product(range(N), repeat=2)), dtype=np.int32)
NUM_POINTS = len(ALL_POINTS)
ALL_INDICES = list(range(NUM_POINTS))

# Precompute squared distances between all pairs
diff = ALL_POINTS[:, np.newaxis, :] - ALL_POINTS[np.newaxis, :, :]
DIST_MATRIX = np.sum(diff * diff, axis=2)  # shape (256, 256)


def _can_add_indices(p_idx, S_indices):
    """
    Check if point p_idx can be added to set S_indices without forming an isosceles triangle.
    Returns True if safe.
    """
    k = len(S_indices)
    if k < 2:
        return True

    S = np.array(S_indices, dtype=np.int32)
    d_ps = DIST_MATRIX[p_idx, S]  # distances from p to each point in S

    # Check 1: p equidistant to two points in S?
    if np.unique(d_ps).size < k:
        return False

    # Distances within S
    D_S = DIST_MATRIX[np.ix_(S, S)]

    # Check 2 & 3: d(p, S[i]) == d(S[i], S[j]) or d(p, S[j]) == d(S[i], S[j])
    # For each pair (i,j), i != j, check if d_ps[i] == D_S[i,j] or d_ps[j] == D_S[i,j]
    # Vectorized:
    # match[i,j] = (d_ps[i] == D_S[i,j]) or (d_ps[j] == D_S[i,j])
    # Exclude diagonal (i==j) where D_S[i,i]=0 and d_ps>0, so no match there.
    match = (d_ps[:, np.newaxis] == D_S) | (d_ps[np.newaxis, :] == D_S)
    # Zero out diagonal to be safe (though it shouldn't matter)
    np.fill_diagonal(match, False)
    if np.any(match):
        return False

    return True


def _build_min_degree_greedy(order_seed=0):
    """
    Greedy construction using minimum-degree heuristic.
    At each step, among all addable points, pick the one that blocks the fewest others.
    """
    rng = random.Random(order_seed)
    S = []
    addable = set(ALL_INDICES)

    while addable:
        candidates = list(addable)
        rng.shuffle(candidates)

        best_p = None
        best_score = float('inf')

        for p in candidates:
            if not _can_add_indices(p, S):
                continue

            # Count how many other addable points would become non-addable
            new_S = S + [p]
            block_count = 0
            for q in addable:
                if q == p:
                    continue
                if not _can_add_indices(q, new_S):
                    block_count += 1

            score = block_count + rng.random() * 0.01
            if score < best_score:
                best_score = score
                best_p = p

        if best_p is None:
            break

        S.append(best_p)
        addable.discard(best_p)

        # Update addable
        new_addable = set()
        for q in addable:
            if _can_add_indices(q, S):
                new_addable.add(q)
        addable = new_addable

    return S


def _is_maximal(S_indices):
    """Check if S_indices is a maximal no-isosceles set (no point can be added)."""
    S_set = set(S_indices)
    for p in ALL_INDICES:
        if p in S_set:
            continue
        if _can_add_indices(p, S_indices):
            return False
    return True


def _remove_and_remmaximize(S_indices, rng, time_limit=5.0):
    """
    Try removing each point from S, then greedily re-maximize.
    Return the smallest maximal set found.
    """
    current = list(S_indices)
    current_size = len(current)
    start = time.time()

    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            if time.time() - start > time_limit:
                return current
            temp = [x for j, x in enumerate(current) if j != i]
            # Re-maximize temp
            S_max = list(temp)
            addable = set(ALL_INDICES) - set(S_max)
            while True:
                added = False
                cands = list(addable)
                rng.shuffle(cands)
                for q in cands:
                    if _can_add_indices(q, S_max):
                        S_max.append(q)
                        addable.discard(q)
                        added = True
                        break
                if not added:
                    break
            if len(S_max) < current_size:
                current = S_max
                current_size = len(S_max)
                improved = True
                break
    return current


def _swap_local_search(S_indices, rng, max_iters=50, time_limit=10.0):
    """
    Try swapping one point in S with one point not in S.
    Accept if the new set is still no-isosceles and maximal, and (optionally) leads to a smaller
    set after re-maximization. For now, accept if valid and maximal.
    """
    current = list(S_indices)
    start = time.time()

    for _ in range(max_iters):
        if time.time() - start > time_limit:
            break
        if not current:
            break

        # Pick a random point to remove
        i = rng.randrange(len(current))
        removed = current[i]
        temp = [x for j, x in enumerate(current) if j != i]

        # Try to find a replacement point not in temp that can be added
        S_set = set(temp)
        replacements = [p for p in ALL_INDICES if p not in S_set]
        rng.shuffle(replacements)

        found = False
        for q in replacements:
            if _can_add_indices(q, temp):
                candidate = temp + [q]
                # Check if candidate is maximal
                if _is_maximal(candidate):
                    current = candidate
                    found = True
                    break

        if not found:
            # Try just removing (if still maximal)
            if _is_maximal(temp):
                current = temp

    return current


def construct():
    start_time = time.time()

    best_set = None
    best_size = float('inf')

    # Baseline: simple greedy in row-major order
    S_row = []
    for p in ALL_INDICES:
        if _can_add_indices(p, S_row):
            S_row.append(p)
    best_set = S_row
    best_size = len(S_row)

    # Multi-start minimum-degree greedy
    num_starts = 60
    for seed in range(num_starts):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break

        S = _build_min_degree_greedy(order_seed=seed)

        # Local search: swap
        rng = random.Random(seed * 1000 + 7)
        S = _swap_local_search(S, rng, max_iters=30, time_limit=8.0)

        # Local search: remove and re-maximize
        S = _remove_and_remmaximize(S, rng, time_limit=5.0)

        if len(S) < best_size:
            best_size = len(S)
            best_set = S

    # Final verification and return
    if best_set is None:
        best_set = [0]

    return [tuple(ALL_POINTS[i]) for i in best_set]


if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
