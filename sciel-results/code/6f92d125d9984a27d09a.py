"""Support-level MIS + Simulated Annealing for kissing number in dimension 11."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    random.seed(42)
    np.random.seed(42)

    # Generate all weight-4 supports (combinations of 4 coordinates from 11)
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # C(11,4) = 330

    # For each support, generate all 16 sign assignments
    sign_patterns = list(itertools.product([1, -1], repeat=4))

    # Build candidate vectors: for each support, 16 sign patterns
    # Total: 330 * 16 = 5280 vectors
    candidates = np.zeros((n_sup * 16, D), dtype=np.int8)
    sup_idx_of_vec = np.zeros(n_sup * 16, dtype=np.int32)

    idx = 0
    for si, sup in enumerate(supports):
        for sp in sign_patterns:
            for pos, s in zip(sup, sp):
                candidates[idx, pos] = s
            sup_idx_of_vec[idx] = si
            idx += 1

    n_vec = len(candidates)

    # Precompute dot products between all vectors
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Norm squared: all weight-4 vectors have norm^2 = 4
    # Constraint: ||v_i - v_j||^2 >= 4 (max norm = 2, so min dist >= 2)
    # => 2*dot[i,j] <= 4 + 4 - 4 = 4
    # => dot[i,j] <= 2
    # Conflict if dot[i,j] > 2

    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Also add weight-2 vectors as Type A (universally compatible with weight-4)
    # Weight-2: norm^2 = 2, max norm in mixed set is 2 (from weight-4)
    # dist^2 >= 4 => 2*dot <= 2 + 4 - 4 = 2 => dot <= 1
    # But actually, we need to check: if we mix w2 and w4, max_norm_sq = 4
    # For two w4: dot <= (4+4-4)/2 = 2
    # For w4 and w2: dot <= (4+2-4)/2 = 1
    # For two w2: dot <= (2+2-4)/2 = 0

    # Let's first solve the w4-only problem, then add w2 vectors that are compatible.

    # Support-level conflict: two supports conflict if there exist sign assignments
    # that are incompatible. But actually, we want to pick at most one vector per support
    # that's compatible with all others.

    # Better approach: work at the vector level but use support structure for speed.
    # Actually, let's do SA directly on the vector selection.

    # Initial solution: greedy min-degree
    x = np.zeros(n_vec, dtype=np.uint8)
    remaining = np.ones(n_vec, dtype=bool)

    while remaining.any() and time.time() < deadline - 60.0:
        rem_indices = np.where(remaining)[0]
        if len(rem_indices) == 0:
            break
        degs = conflict[rem_indices].sum(axis=1)
        min_local = np.argmin(degs)
        new_idx = rem_indices[min_local]
        x[new_idx] = 1
        remaining[new_idx] = False
        neighbors = conflict[new_idx].astype(bool)
        remaining[neighbors] = False

    # Simulated Annealing on the selection
    # Objective: maximize number of selected vectors subject to no conflicts
    # Move: flip a bit (add or remove a vector)

    def get_violations(sel_mask):
        """Return count of conflicts in the selected set."""
        if sel_mask.sum() < 2:
            return 0
        sub_conflict = conflict[np.ix_(sel_mask, sel_mask)]
        return sub_conflict.sum() // 2

    def get_conflict_mask(sel_mask):
        """For each vertex, return whether it conflicts with any selected vertex."""
        if sel_mask.sum() == 0:
            return np.zeros(n_vec, dtype=bool)
        return (conflict[:, sel_mask].sum(axis=1) > 0)

    current_x = x.copy()
    current_violations = get_violations(current_x)
    current_size = current_x.sum()

    best_x = current_x.copy()
    best_size = current_size

    # SA parameters
    T = 5.0
    cooling = 0.9995
    min_T = 0.01

    while T > min_T and time.time() < deadline - 10.0:
        # Random move: pick a random vertex and flip it
        idx = random.randint(0, n_vec - 1)

        if current_x[idx] == 1:
            # Remove vertex idx
            new_x = current_x.copy()
            new_x[idx] = 0
            # New violations: need to recompute
            # Faster: the removed vertex was contributing to some conflicts
            # Let's just recompute for simplicity (n_vec=5280, this is O(n^2) worst case)
            # But we can be smarter:
            new_size = current_size - 1
            # Check if any remaining selected vertices now have conflicts resolved
            # Actually, removing a vertex can only reduce violations
            # New violations = old violations - (conflicts involving idx)
            conflicts_with_idx = conflict[idx, current_x].sum()
            # But some of those conflicts might be double-counted... 
            # Let's just recompute
            new_violations = get_violations(new_x)
            delta_v = new_violations - current_violations
            delta_s = new_size - current_size
        else:
            # Add vertex idx
            # Check if idx conflicts with any currently selected
            conflicts_with_selected = conflict[idx, current_x].sum()
            if conflicts_with_selected > 0:
                # This would add conflicts
                new_x = current_x.copy()
                new_x[idx] = 1
                new_size = current_size + 1
                new_violations = get_violations(new_x)
                delta_v = new_violations - current_violations
                delta_s = new_size - current_size
            else:
                # No new conflicts, pure improvement
                new_x = current_x.copy()
                new_x[idx] = 1
                new_size = current_size + 1
                new_violations = current_violations
                delta_v = 0
                delta_s = 1

        # Acceptance: maximize (size - violations * penalty)
        # We want to minimize violations first, then maximize size
        # Use: accept if delta_v < 0, or if delta_v == 0 and delta_s > 0, or with probability
        if delta_v < 0 or (delta_v == 0 and delta_s > 0):
            current_x = new_x
            current_violations = new_violations
            current_size = new_size
            if current_size > best_size:
                best_x = current_x.copy()
                best_size = current_size
        elif delta_v == 0 and delta_s < 0:
            # Removing a vertex with no conflicts - bad
            pass
        else:
            # delta_v > 0 or (delta_v == 0 and delta_s <= 0)
            # Accept with probability exp(-delta / T)
            # We want to minimize violations, so delta_v > 0 is bad
            # Use: probability = exp(-delta_v / T) if delta_v > 0
            if delta_v > 0:
                prob = np.exp(-delta_v / T)
            else:
                prob = 0.5
            if random.random() < prob:
                current_x = new_x
                current_violations = new_violations
                current_size = new_size
                if current_violations == 0 and current_size > best_size:
                    best_x = current_x.copy()
                    best_size = current_size

        T *= cooling

    # Use best solution found
    x = best_x

    # Final cleanup: remove any vertices that cause conflicts
    # Iterate until stable
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for i in np.where(x == 1)[0]:
            if conflict[i, x].sum() > 0:
                x[i] = 0
                changed = True

    # Add weight-2 vectors that are compatible
    w2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int8)
            for pos, s in zip(positions, signs):
                v[pos] = s
            w2_candidates.append(v)

    w2_arr = np.array(w2_candidates, dtype=np.int8)
    n_w2 = len(w2_arr)

    # Check compatibility of w2 vectors with current w4 selection
    # For w2 v and w4 u: dot <= 1
    # For w2 v and w2 w: dot <= 0

    if n_w2 > 0 and x.sum() > 0:
        sel_w4 = candidates[x.astype(bool)]
        # dot between w2 and selected w4
        dot_w2_w4 = w2_arr.astype(np.int32) @ sel_w4.T.astype(np.int32)
        # w2 compatible with all selected w4 if max dot <= 1
        compat_with_w4 = (dot_w2_w4.max(axis=1) <= 1)

        # Among compatible w2, find a large independent set (dot <= 0 between w2s)
        dot_w2_w2 = w2_arr.astype(np.int32) @ w2_arr.T.astype(np.int32)
        conflict_w2 = (dot_w2_w2 > 0).astype(np.uint8)
        np.fill_diagonal(conflict_w2, 0)

        # Greedy on compatible w2
        remaining_w2 = compat_with_w4.copy()
        sel_w2 = []
        while remaining_w2.any() and time.time() < deadline - 2.0:
            rem_idx = np.where(remaining_w2)[0]
            if len(rem_idx) == 0:
                break
            degs_w2 = conflict_w2[rem_idx].sum(axis=1)
            min_local = np.argmin(degs_w2)
            new_idx = rem_idx[min_local]
            sel_w2.append(new_idx)
            remaining_w2[new_idx] = False
            neighbors_w2 = conflict_w2[new_idx].astype(bool)
            remaining_w2[neighbors_w2] = False

        # Combine
        result = np.vstack([sel_w4, w2_arr[sel_w2]])
    else:
        result = candidates[x.astype(bool)]

    if len(result) == 0:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    return result.astype(np.int64)
