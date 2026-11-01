import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Generate all candidate vectors with norm² in {2, 4}
    candidates = []

    # Type A: weight-2 vectors in {-1,0,1}^11 (norm² = 2)
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)

    # Type B: weight-1 vectors in {-2,0,2}^11 (norm² = 4)
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)

    # Type C: weight-4 vectors in {-1,0,1}^11 (norm² = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)

    if n == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)

    # Conflict: dist² < 4 (i.e., ||u-v||² < 4)
    # dist² = ||u||² + ||v||² - 2*u·v
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = (dist_sq < 4).astype(np.int8)
    np.fill_diagonal(conflicts, 0)

    # Convert to boolean for faster operations
    conflict_bool = conflicts.astype(bool)

    # Strategy 1: Greedy minimum degree with multiple random seeds
    best_selected = None
    best_count = 0

    def greedy_min_degree(seed):
        rng = np.random.default_rng(seed)
        alive = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)

        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]

            selected[chosen] = True
            to_remove = np.where(conflict_bool[chosen] | (np.arange(n) == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)

        return selected, selected.sum()

    # Run greedy with several seeds
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        sel, cnt = greedy_min_degree(seed)
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()

    # Strategy 2: Spectral initialization + greedy
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        degree = conflicts.sum(axis=1).astype(np.float64)
        L = np.diag(degree) - conflicts.astype(np.float64)
        L_sparse = csr_matrix(L)

        eigvals, eigvecs = eigsh(L_sparse, k=4, sigma=0, which='LM')
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Use Fiedler vector for ordering
        fiedler = eigvecs[:, 1]
        order = np.argsort(fiedler)

        alive = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)
        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflict_bool[i] | (np.arange(n) == i))[0]
            alive[to_remove] = False

        cnt = selected.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = selected.copy()
    except Exception:
        pass

    # Strategy 3: Local search refinement (1-move add, 2-swap)
    if best_selected is not None:
        selected = best_selected.copy()
        improved = True
        max_rounds = 200
        round_num = 0

        while improved and round_num < max_rounds:
            if time.time() > deadline - 15:
                break
            improved = False
            round_num += 1

            non_selected_indices = np.where(~selected)[0]
            selected_indices = np.where(selected)[0]

            if len(non_selected_indices) == 0:
                break

            # Try to add any non-selected vertex with 0 conflicts with selected
            added_any = False
            for v in non_selected_indices:
                if time.time() > deadline - 15:
                    break
                sel_neighbors = np.where(conflict_bool[v] & selected)[0]
                if len(sel_neighbors) == 0:
                    selected[v] = True
                    improved = True
                    added_any = True
                    break

            if added_any:
                continue

            # Try 2-swap: remove one selected u, add one non-selected v
            # where v conflicts with exactly one selected (u), and removing u
            # allows v to be added, and net gain is >= 0 (we'll check if it helps)
            # Actually for count improvement we need: remove u, add v, and v
            # conflicts with only u among selected. Net change = 0, but might
            # enable future additions. Let's try swaps that don't decrease count
            # and hope they enable further improvements.

            best_swap_gain = -1
            best_swap_u = -1
            best_swap_v = -1

            for v in non_selected_indices:
                if time.time() > deadline - 15:
                    break
                sel_neighbors = np.where(conflict_bool[v] & selected)[0]
                if len(sel_neighbors) == 1:
                    u = sel_neighbors[0]
                    # If we remove u and add v: net 0 change in count
                    # But v might conflict with fewer other non-selected vertices
                    # Let's compute: after swap, how many new non-selected can we add?
                    # This is expensive; just do the swap and see if next round adds
                    # For now, accept 0-gain swaps randomly
                    if rng is None:
                        import numpy as _np
                        rng = _np.random.default_rng(0)
                    if rng.random() < 0.1:
                        best_swap_u = u
                        best_swap_v = v
                        best_swap_gain = 0
                        break

            if best_swap_v >= 0:
                selected[best_swap_u] = False
                selected[best_swap_v] = True
                improved = True

            # Also try: remove a selected vertex that has many conflicts,
            # to free up space for multiple non-selected vertices
            # This is a heuristic: find selected vertex u with max degree to non-selected
            # Remove it, then try to add as many non-selected as possible
            if not improved:
                for u in selected_indices:
                    if time.time() > deadline - 15:
                        break
                    # Count non-selected vertices that conflict only with u among selected
                    ns = np.where(~selected)[0]
                    if len(ns) == 0:
                        break
                    # For each non-selected v, check if its only conflict with selected is u
                    # This is O(n) per u, too slow. Skip for now.
                    pass

    # Strategy 4: More random restarts with different orderings
    for restart in range(30):
        if time.time() > deadline - 10:
            break

        rng2 = np.random.default_rng(1000 + restart)
        alive = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)

        # Random ordering
        order = rng2.permutation(n)

        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflict_bool[i] | (np.arange(n) == i))[0]
            alive[to_remove] = False

        cnt = selected.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = selected.copy()

    # Final local search polish on the best solution
    if best_selected is not None:
        selected = best_selected.copy()

        # Repeatedly add any vertex with no conflicts until none can be added
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            changed = False
            non_selected_indices = np.where(~selected)[0]
            for v in non_selected_indices:
                sel_neighbors = np.where(conflict_bool[v] & selected)[0]
                if len(sel_neighbors) == 0:
                    selected[v] = True
                    changed = True

            if not changed:
                break

        if selected.sum() > best_count:
            best_count = selected.sum()
            best_selected = selected.copy()

    result = candidates[best_selected]
    return result
