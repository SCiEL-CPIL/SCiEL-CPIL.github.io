"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
TOTAL = N * N

# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _build_tables():
    """Precompute distance-squared matrix and isosceles triple structure."""
    points = list(itertools.product(range(N), repeat=2))
    idx = {p: i for i, p in enumerate(points)}
    
    # Distance squared matrix
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)
    
    # For each point, precompute which pairs (i,j) form isosceles with it
    # iso_with[p] = set of (i, j) such that {p, i, j} is isosceles
    # Actually, we want: for a candidate point c, and a set S, 
    # c cannot be added if there exist i,j in S such that {c,i,j} is isosceles.
    # So for each c, precompute the set of pairs (i,j) that form isosceles with c.
    
    # iso_pairs[c] = list of (i, j) with i < j such that {c, i, j} is isosceles
    iso_pairs = [None] * TOTAL
    for c in range(TOTAL):
        pairs = []
        for i in range(TOTAL):
            if i == c:
                continue
            for j in range(i + 1, TOTAL):
                if j == c:
                    continue
                d_ci = dist2[c, i]
                d_cj = dist2[c, j]
                d_ij = dist2[i, j]
                if d_ci == d_cj or d_ci == d_ij or d_cj == d_ij:
                    pairs.append((i, j))
        iso_pairs[c] = pairs
    
    return points, idx, dist2, iso_pairs


def _can_add_fast(c, chosen_set, iso_pairs):
    """Check if point c can be added to chosen_set (as a set of indices)."""
    # c can be added iff for all (i,j) in iso_pairs[c], NOT (i in chosen_set AND j in chosen_set)
    for (i, j) in iso_pairs[c]:
        if i in chosen_set and j in chosen_set:
            return False
    return True


def _is_maximal_fast(set_indices, iso_pairs, deadline):
    """Check if set of indices is a maximal no-isosceles set."""
    set_set = set(set_indices)
    for c in range(TOTAL):
        if c in set_set:
            continue
        # If c can be added, not maximal
        if _can_add_fast(c, set_set, iso_pairs):
            return False
    return True


def _greedy_maximal_fast(order, iso_pairs, deadline):
    """Greedy maximal construction given an ordering of point indices."""
    chosen = []
    chosen_set = set()
    for c in order:
        if time.time() > deadline:
            return None
        if _can_add_fast(c, chosen_set, iso_pairs):
            chosen.append(c)
            chosen_set.add(c)
    return chosen


def _prune_fast(set_indices, iso_pairs, deadline):
    """Try to remove points while maintaining maximality."""
    set_list = list(set_indices)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(set_list) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = set_list[:i] + set_list[i + 1:]
            if _is_maximal_fast(candidate, iso_pairs, deadline):
                set_list = candidate
                improved = True
                break
    return set_list


def _local_search(set_indices, iso_pairs, deadline, rng):
    """Local search: try swaps and removals to minimize size."""
    current = list(set_indices)
    current_set = set(current)
    best = current[:]
    best_size = len(current)
    
    # First try removals (like prune)
    for i in range(len(current) - 1, -1, -1):
        if time.time() > deadline:
            break
        candidate = current[:i] + current[i + 1:]
        if _is_maximal_fast(candidate, iso_pairs, deadline):
            current = candidate
            current_set = set(candidate)
            if len(current) < best_size:
                best = current[:]
                best_size = len(current)
    
    # Now try swaps: remove one, add another (net zero change in size)
    # This can enable subsequent removals
    for iteration in range(50):
        if time.time() > deadline:
            break
        improved = False
        # Try all pairs of (remove a, add b)
        for a_idx in range(len(current)):
            if time.time() > deadline:
                break
            a = current[a_idx]
            # Try adding each point not in current
            for b in range(TOTAL):
                if b in current_set:
                    continue
                # Check if we can swap: remove a, add b
                # New set = current - {a} + {b}
                # Must be valid (no isosceles within new set) and maximal
                new_set = set(current)
                new_set.remove(a)
                new_set.add(b)
                
                # Check validity: no isosceles triple within new_set
                valid = True
                for c in new_set:
                    for (i, j) in iso_pairs[c]:
                        if i in new_set and j in new_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check maximality
                if _is_maximal_fast(new_set, iso_pairs, deadline):
                    current = list(new_set)
                    current_set = new_set
                    improved = True
                    if len(current) < best_size:
                        best = current[:]
                        best_size = len(current)
                    break
            if improved:
                break
        if not improved:
            break
    
    return best


def construct():
    deadline = time.time() + 280
    points, idx, dist2, iso_pairs = _build_tables()
    all_indices = list(range(TOTAL))
    
    best = None
    best_size = float('inf')
    
    # First: deterministic row-major greedy
    result = _greedy_maximal_fast(all_indices, iso_pairs, deadline)
    if result is not None:
        result = _prune_fast(result, iso_pairs, deadline)
        result = _local_search(result, iso_pairs, deadline, random.Random(42))
        best = result
        best_size = len(result)
    
    # Randomized restarts
    rng = random.Random(42)
    for trial in range(5000):
        if time.time() > deadline:
            break
        order = all_indices[:]
        rng.shuffle(order)
        result = _greedy_maximal_fast(order, iso_pairs, deadline)
        if result is None:
            break
        
        result = _prune_fast(result, iso_pairs, deadline)
        result = _local_search(result, iso_pairs, deadline, rng)
        
        if len(result) < best_size:
            best = result
            best_size = len(result)
            if best_size <= 10:
                break
    
    # Convert indices back to points
    if best is not None:
        return [points[i] for i in best]
    return []
# EVOLVE-BLOCK-END
