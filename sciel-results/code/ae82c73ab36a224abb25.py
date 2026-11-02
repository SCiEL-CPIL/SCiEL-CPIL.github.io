import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

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

    # We assume the set will contain at least one vector with norm sq 4.
    # If not, R^2 would be 3, but Type 1/2 are very packable, so this is unlikely to be optimal.
    # Even if R^2=3, the constraints for Type 3 vs Type 3 are the same (dot <= 1).
    # Constraints for Type 1/2 vs Type 1/2 would be looser (dot <= 2.5 -> 2), same as R^2=4.
    # Constraints for Type 1/2 vs Type 3 would be looser (dot <= 1.5 -> 1), same as R^2=4.
    # So using R^2=4 is a safe conservative bound that yields valid solutions even if R^2 turns out to be 3.
    R_sq = 4

    # Threshold matrix: Max allowed dot product for pair (i,j)
    # 2 * M[i,j] <= norms_sq[i] + norms_sq[j] - R_sq
    # M[i,j] <= (norms_sq[i] + norms_sq[j] - R_sq) / 2
    # Since M is integer, we can use floor division by 2.
    num = norms_sq[:, None] + norms_sq[None, :] - R_sq
    Thresh = num // 2

    # Precompute conflict matrix for efficiency: conflict[i,j] = True if M[i,j] > Thresh[i,j]
    # This is O(N^2) but vectorized.
    conflict = (M > Thresh).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Conflict counts (number of other vectors that conflict with i)
    conflict_counts = np.sum(conflict, axis=1)
    # Compatibility counts (number of other vectors that are compatible with i)
    compat_counts = N - 1 - conflict_counts

    best_set = None
    best_count = 0

    rng = np.random.default_rng(42)

    # Strategy A: Random orderings
    for trial in range(50):
        if time.time() > deadline:
            break

        if trial == 0:
            order = np.arange(N)
        else:
            order = rng.permutation(N)

        selected_mask = np.zeros(N, dtype=bool)
        count = 0

        for idx in order:
            if count == 0:
                selected_mask[idx] = True
                count += 1
                continue

            # Check compatibility with all selected
            # conflict[idx, selected_mask] gives a boolean array where True means conflict
            if not np.any(conflict[idx, selected_mask]):
                selected_mask[idx] = True
                count += 1

        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()

    # Strategy B: Greedy with "least conflicts" priority
    if time.time() < deadline - 20:
        order_b = np.argsort(conflict_counts, kind='stable')

        selected_mask = np.zeros(N, dtype=bool)
        count = 0

        for idx in order_b:
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

    # Strategy C: Greedy with "most central" (highest degree in compatibility graph)
    if time.time() < deadline - 10:
        order_c = np.argsort(-compat_counts, kind='stable')

        selected_mask = np.zeros(N, dtype=bool)
        count = 0

        for idx in order_c:
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

    # Strategy D: Local improvement - try to add unselected vectors
    if best_set is not None and time.time() < deadline - 5:
        unselected = np.where(~best_set)[0]
        # Sort unselected by conflict count to try "easy" ones first? Or just iterate.
        # Since we already did a pass, let's just iterate and add if possible.
        for idx in unselected:
            if time.time() > deadline - 2:
                break
            if not np.any(conflict[idx, best_set]):
                best_set[idx] = True
                best_count += 1

    if best_set is None:
        # Fallback: just the basis vectors
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return candidates[best_set].astype(np.int64)
