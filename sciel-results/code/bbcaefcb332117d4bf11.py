import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Generate candidates
    candidates = []

    # Type 1: One 2, rest 0: ±2e_i (Norm sq 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l (Norm sq 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    # Type 3: Three 1s, rest 0: ±e_i ± e_j ± e_k (Norm sq 3)
    for idxs in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix M
    M = candidates @ candidates.T

    # Compute squared norms
    norms_sq = np.sum(candidates * candidates, axis=1)

    R_sq = 4

    # Threshold matrix: Max allowed dot product for pair (i,j)
    num = norms_sq[:, None] + norms_sq[None, :] - R_sq
    Thresh = num // 2

    # Conflict matrix: True if pair (i,j) conflicts
    conflict = M > Thresh
    # Zero out diagonal (self-conflict doesn't matter for independent set)
    np.fill_diagonal(conflict, False)

    # Build adjacency as bitsets for fast operations
    # For N up to ~5000, we can use numpy boolean arrays
    # But for branch-and-bound, we need efficient operations

    # First, let's try to get a good initial solution with greedy
    best_set = None
    best_count = 0

    # Greedy strategy: sort by degree (number of conflicts), pick lowest degree first
    conflict_counts = np.sum(conflict, axis=1)
    order = np.argsort(conflict_counts, kind='stable')

    selected_mask = np.zeros(N, dtype=bool)
    count = 0

    for idx in order:
        if count == 0:
            selected_mask[idx] = True
            count += 1
            continue

        if not np.any(conflict[idx, selected_mask]):
            selected_mask[idx] = True
            count += 1

    best_count = count
    best_set = selected_mask.copy()

    # Try several random orderings
    rng = np.random.default_rng(42)
    for trial in range(100):
        if time.time() > deadline - 60:
            break

        order_r = rng.permutation(N)
        selected_mask = np.zeros(N, dtype=bool)
        count = 0

        for idx in order_r:
            if count == 0:
                selected_mask[idx] = True
                count += 1
                continue

            if not np.any(conflict[idx, selected_mask]):
                selected_mask[idx] = True
                count += 1

        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()

    # Now try branch-and-bound to improve
    # Use a simple approach: try to add vertices from unselected set
    # For each unselected vertex, check if it can be added
    # This is a local search / greedy improvement

    # More sophisticated: Use a maximum independent set algorithm on the conflict graph
    # Since N can be ~5000, we need an efficient approach

    # Let's use a greedy coloring bound approach
    # First, identify the subgraph of unselected vertices
    unselected = np.where(~best_set)[0]

    if len(unselected) > 0 and time.time() < deadline - 10:
        # Try to greedily add more vertices from unselected
        # Sort unselected by their conflict count within the current set
        for idx in unselected:
            if time.time() > deadline - 5:
                break
            if not np.any(conflict[idx, best_set]):
                best_set[idx] = True
                best_count += 1

    # Try multiple passes of improvement
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        improved = False
        unselected = np.where(~best_set)[0]
        for idx in unselected:
            if time.time() > deadline - 5:
                break
            if not np.any(conflict[idx, best_set]):
                best_set[idx] = True
                best_count += 1
                improved = True
        if not improved:
            break

    # Now try a more sophisticated branch-and-bound
    # Focus on the subgraph induced by unselected vertices
    # Try to find additional independent sets in this subgraph

    if time.time() < deadline - 20:
        unselected = np.where(~best_set)[0]
        if len(unselected) > 0:
            # Build the subgraph on unselected vertices
            sub_conflict = conflict[np.ix_(unselected, unselected)]
            sub_N = len(unselected)

            # Greedy coloring for upper bound
            # Use a simple greedy coloring
            color = -np.ones(sub_N, dtype=np.int32)
            num_colors = 0

            for i in range(sub_N):
                if color[i] >= 0:
                    continue
                # Try to assign a color
                used_colors = set()
                for j in range(sub_N):
                    if sub_conflict[i, j] and color[j] >= 0:
                        used_colors.add(color[j])

                c = 0
                while c in used_colors:
                    c += 1
                color[i] = c
                num_colors = max(num_colors, c + 1)

            # Upper bound on independent set size is num_colors
            # If num_colors <= 0, no improvement possible
            if num_colors > 0:
                # Try greedy on the subgraph
                sub_conflict_counts = np.sum(sub_conflict, axis=1)
                sub_order = np.argsort(sub_conflict_counts, kind='stable')

                sub_selected = np.zeros(sub_N, dtype=bool)
                sub_count = 0

                for idx in sub_order:
                    if sub_count == 0:
                        sub_selected[idx] = True
                        sub_count += 1
                        continue

                    if not np.any(sub_conflict[idx, sub_selected]):
                        sub_selected[idx] = True
                        sub_count += 1

                if sub_count > 0:
                    # Add these to best_set
                    for i, idx in enumerate(unselected):
                        if sub_selected[i]:
                            best_set[idx] = True
                            best_count += 1

    # Final pass to try to add any remaining vertices
    if time.time() < deadline - 2:
        unselected = np.where(~best_set)[0]
        for idx in unselected:
            if time.time() > deadline - 1:
                break
            if not np.any(conflict[idx, best_set]):
                best_set[idx] = True
                best_count += 1

    if best_set is None:
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return candidates[best_set].astype(np.int64)
