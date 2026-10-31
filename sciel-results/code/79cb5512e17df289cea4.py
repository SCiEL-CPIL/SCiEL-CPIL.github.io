import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Convert to sparse for efficiency
    from scipy.sparse import csr_matrix
    conflicts_sparse = csr_matrix(conflicts)

    best_selected = None
    best_count = 0

    # Strategy 1: Greedy minimum degree with multiple random seeds
    for seed in [42, 123, 456, 789, 101]:
        if time.time() > deadline - 30:
            break

        rng = np.random.default_rng(seed)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        # Compute current degrees
        degrees = conflicts.sum(axis=1).astype(np.int64).copy()

        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]

            selected[chosen] = True
            # Remove chosen and its neighbors from alive
            to_remove = np.where(conflicts[chosen] | (np.arange(N4) == chosen))[0]
            alive[to_remove] = False
            # Update degrees for remaining alive vertices
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                # For each remaining vertex, count how many of its neighbors are in to_remove
                # This is the number of neighbors that just died
                deg_updates = conflicts[remaining][:, to_remove].sum(axis=1)
                degrees[remaining] -= deg_updates

        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    # Strategy 2: Spectral initialization + greedy
    from scipy.sparse.linalg import eigsh
    L = np.diag(conflicts.sum(axis=1)) - conflicts.astype(np.float64)
    L_sparse = csr_matrix(L)

    try:
        eigvals, eigvecs = eigsh(L_sparse, k=3, sigma=0, which='LM')
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
    except Exception:
        rng = np.random.default_rng(42)
        fiedler = rng.standard_normal(N4)

    # Order by fiedler value
    order = np.argsort(fiedler)
    alive = np.ones(N4, dtype=bool)
    selected = np.zeros(N4, dtype=bool)

    for i in order:
        if not alive[i]:
            continue
        selected[i] = True
        to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
        alive[to_remove] = False

    count = selected.sum()
    if count > best_count:
        best_count = count
        best_selected = selected.copy()

    # Strategy 3: Local search refinement - try to add vertices with no conflicts
    if best_selected is not None:
        selected = best_selected.copy()
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            non_selected = np.where(~selected)[0]
            if len(non_selected) == 0:
                break
            changed = False
            for v in non_selected:
                if np.any(conflicts[v] & selected):
                    continue
                selected[v] = True
                changed = True
            if not changed:
                break

        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    # Strategy 4: More random restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break

        rng2 = np.random.default_rng(1000 + restart * 7)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        order = rng2.permutation(N4)

        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
            alive[to_remove] = False

        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    result_v4 = vectors_v4[best_selected]

    # Add all V1 vectors
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors
