import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 105.0

    # Generate all weight-4 vectors in R^11
    # C(11,4) * 2^4 = 330 * 16 = 5280
    indices = range(D)
    candidates = []
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(comb, signs):
                v[idx] = val
            candidates.append(v)

    V = np.array(candidates, dtype=np.int32)
    N = V.shape[0]

    if N == 0:
        # Fallback: weight-2 vectors
        cands2 = []
        for comb in itertools.combinations(indices, 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for idx, val in zip(comb, signs):
                    v[idx] = val
                cands2.append(v)
        return np.array(cands2, dtype=np.int64)

    # Precompute the Gram matrix: M[i,j] = V[i] . V[j]
    # Max value is 4, min is -4. int32 is safe.
    M = V @ V.T

    # Conflict condition: |V[i] - V[j]|^2 < max_norm^2
    # max_norm^2 = 4 (since all weight-4 vectors have norm^2 = 4)
    # |V[i] - V[j]|^2 = |V[i]|^2 + |V[j]|^2 - 2 V[i].V[j] = 4 + 4 - 2 M[i,j] = 8 - 2 M[i,j]
    # We need 8 - 2 M[i,j] >= 4  =>  4 >= 2 M[i,j]  =>  M[i,j] <= 2
    # So two vectors are compatible iff M[i,j] <= 2.
    # Note: M[i,i] = 4, so a vector is always in conflict with itself (which is fine for independent set).

    # Build conflict matrix: conflict[i,j] = True if M[i,j] > 2
    conflict = M > 2

    # We want a large independent set in the graph defined by 'conflict'.
    # Equivalent: large clique in the compatibility graph (where edge exists if M[i,j] <= 2 and i != j).

    # Strategy: Greedy with random restarts, but with a better ordering heuristic.
    # For each restart, we order vertices by a random tie-breaking on top of degree.
    # Or: use a "largest first" heuristic with randomization.

    best_indices = None
    best_count = 0

    # Precompute degrees for initial ordering
    degrees = np.sum(conflict, axis=1)

    # Multiple strategies:
    # 1. Random permutation greedy
    # 2. Degree-based greedy (pick lowest degree first)
    # 3. Random + degree hybrid

    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1

        # Hybrid ordering: sort by (random noise + degree) to break ties
        # Actually, let's try a few different strategies per restart.
        strategy = num_restarts % 3

        if strategy == 0:
            # Pure random
            order = np.random.permutation(N)
        elif strategy == 1:
            # Sort by degree (ascending), random tie-break
            order = np.lexsort((np.random.rand(N), degrees))
        else:
            # Randomized: add small noise to degree
            noisy_deg = degrees.astype(float) + np.random.rand(N) * 0.1
            order = np.argsort(noisy_deg)

        active = np.ones(N, dtype=bool)
        selected = []

        for idx in order:
            if not active[idx]:
                continue
            selected.append(idx)
            # Remove this vertex and all its neighbors from active
            active[idx] = False
            # neighbors are where conflict[idx] is True
            neighbors = np.where(conflict[idx])[0]
            if len(neighbors) > 0:
                active[neighbors] = False

        count = len(selected)
        if count > best_count:
            best_count = count
            best_indices = np.array(selected, dtype=int)
            # Check time
            if time.time() > deadline:
                break

    # Additional local improvement: try to add more vertices by checking compatibility
    # This is a simple 1-step local search
    if best_indices is not None and best_count > 0:
        selected_set = set(best_indices.tolist())
        # Try to add any vertex not in selected_set that is compatible with all in selected_set
        for candidate_idx in range(N):
            if candidate_idx in selected_set:
                continue
            # Check if candidate is compatible with all selected
            compat = True
            for sel_idx in best_indices:
                if M[candidate_idx, sel_idx] > 2:
                    compat = False
                    break
            if compat:
                best_indices = np.append(best_indices, candidate_idx)
                best_count += 1
                selected_set.add(candidate_idx)

    if best_indices is None or len(best_indices) == 0:
        best_indices = np.array([0], dtype=int)

    result = V[best_indices]
    return result.astype(np.int64)
