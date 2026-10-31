import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Generate all weight-2 vectors in {-1, 0, 1}^11 (Norm^2 = 2)
    # These are automatically compatible with each other when in disjoint coordinate planes
    # and with weight-1 vectors (±2 in single coordinate)
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_candidates.append(v)

    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v2 = np.array(v2_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N2 = len(vectors_v2)

    if N2 == 0:
        return vectors_v1

    # For weight-2 vectors, conflict if dot product >= 1 (i.e., they share a coordinate with same sign)
    # dot(v_i, v_j) >= 1 means they share at least one coordinate with the same sign
    # This makes distance^2 = 2 + 2 - 2*dot < 2 = max_norm^2, violating the constraint
    dots = vectors_v2 @ vectors_v2.T
    conflicts = dots >= 1
    np.fill_diagonal(conflicts, False)

    # Convert conflicts to int for efficient operations
    conflicts_int = conflicts.astype(np.int8)

    # Get Fiedler vector for spectral initialization
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    degree = conflicts.sum(axis=1)
    L = np.diag(degree) - conflicts.astype(np.float64)
    L_sparse = csr_matrix(L)

    try:
        eigvals, eigvecs = eigsh(L_sparse, k=5, sigma=0, which='LM')
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
    except Exception:
        rng = np.random.default_rng(42)
        fiedler = rng.standard_normal(N2)

    best_selected = None
    best_count = 0

    def add_free_vertices(selected, conflicts_int, N):
        """Add all vertices that don't conflict with any selected vertex."""
        changed = True
        while changed:
            changed = False
            non_sel = np.where(~selected)[0]
            for v in non_sel:
                if np.any(conflicts_int[v] & selected):
                    continue
                selected[v] = True
                changed = True
        return selected

    def greedy_min_degree(conflicts, N, seed=42):
        """Greedy minimum degree with incremental updates."""
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)
        rng = np.random.default_rng(seed)

        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]

            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (np.arange(N) == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)

        return selected

    def greedy_ordering(conflicts, N, order):
        """Greedy selection with a given vertex ordering."""
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)

        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(N) == i))[0]
            alive[to_remove] = False

        return selected

    def kempe_chain_improve(selected, conflicts, N, max_iters=5000, time_limit=None):
        """
        Vectorized Kempe-chain-like improvement for independent set.
        Tries swaps: remove selected u, add non-selected v, then greedily add all free vertices.
        """
        selected = selected.copy()
        current_count = selected.sum()

        if time_limit is None:
            time_limit = deadline - 10

        for iteration in range(max_iters):
            if time.time() > time_limit:
                break

            improved = False
            best_swap = None
            best_gain = 0

            sel_idx = np.where(selected)[0]
            non_sel_idx = np.where(~selected)[0]

            if len(non_sel_idx) == 0 or len(sel_idx) == 0:
                break

            # Vectorized: for each non-selected vertex, count selected neighbors
            # conflicts is (N, N) bool matrix
            sel_neigh_counts = (conflicts[non_sel_idx] & selected[None, :]).sum(axis=1)

            # Find vertices with exactly 1 selected neighbor
            single_neigh_mask = sel_neigh_counts == 1
            if not np.any(single_neigh_mask):
                # Try 1-flip: remove a selected vertex, add all free
                if len(sel_idx) > 0:
                    # For each selected vertex, count how many non-selected it conflicts with
                    # Removing u frees up all non-selected vertices that only conflict with u among selected
                    # Approximate: just try removing each selected vertex and see net gain
                    # Use vectorized approach: for each u in sel_idx, compute gain
                    # gain(u) = (number of non-selected w that have no selected neighbor after removing u) - 1
                    # A non-selected w has no selected neighbor after removing u iff
                    # all its selected neighbors are {u} (i.e., it has exactly 1 selected neighbor which is u)
                    # OR it has 0 selected neighbors (already free, would be added regardless)

                    # Actually, let's just do a few random removals
                    rng_local = np.random.default_rng(iteration)
                    trial_indices = rng_local.choice(len(sel_idx), size=min(5, len(sel_idx)), replace=False)
                    for ti in trial_indices:
                        u = sel_idx[ti]
                        selected[u] = False
                        # Add all free vertices
                        non_sel_now = np.where(~selected)[0]
                        # Vectorized: which non-selected have no conflict with selected?
                        if len(non_sel_now) > 0:
                            free_mask = ~(conflicts[non_sel_now] & selected[None, :]).any(axis=1)
                            to_add = non_sel_now[free_mask]
                            if len(to_add) > 0:
                                selected[to_add] = True
                        gain = selected.sum() - current_count
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = ('remove', u, to_add if len(to_add) > 0 else np.array([], dtype=np.int64))
                        # Revert
                        selected[u] = True
                        if len(to_add) > 0:
                            selected[to_add] = False

                    if best_swap is not None and best_gain > 0:
                        kind, u, to_add = best_swap
                        selected[u] = False
                        if len(to_add) > 0:
                            selected[to_add] = True
                        current_count = selected.sum()
                        improved = True
                if not improved:
                    break
                continue

            # Get the single selected neighbor for each candidate
            # For vertices with exactly 1 selected neighbor, find which one
            candidates = non_sel_idx[single_neigh_mask]
            # For each candidate v, the single selected neighbor is:
            # sel_neigh = np.where(conflicts[v] & selected)[0][0]
            # Vectorized:
            # conflicts[candidates] is (len(candidates), N)
            # We need the column index where selected is True
            # Since each has exactly 1, we can use argmax on the masked row
            cand_sel_conflicts = conflicts[candidates] & selected[None, :]
            # Each row has exactly one True
            neighbor_indices = np.argmax(cand_sel_conflicts, axis=1)

            # Try each swap (or a random subset for speed)
            n_cands = len(candidates)
            if n_cands > 100:
                rng_local = np.random.default_rng(iteration)
                trial_cand_indices = rng_local.choice(n_cands, size=100, replace=False)
            else:
                trial_cand_indices = np.arange(n_cands)

            for ci in trial_cand_indices:
                v = candidates[ci]
                u = neighbor_indices[ci]

                # Tentative swap
                selected[u] = False
                selected[v] = True

                # Add all free vertices (vectorized)
                non_sel_now = np.where(~selected)[0]
                if len(non_sel_now) > 0:
                    free_mask = ~(conflicts[non_sel_now] & selected[None, :]).any(axis=1)
                    to_add = non_sel_now[free_mask]
                    if len(to_add) > 0:
                        selected[to_add] = True

                gain = selected.sum() - current_count

                if gain > best_gain:
                    best_gain = gain
                    best_swap = (u, v, to_add if len(to_add) > 0 else np.array([], dtype=np.int64))

                # Revert
                selected[u] = True
                selected[v] = False
                if len(to_add) > 0:
                    selected[to_add] = False

            if best_swap is not None and best_gain > 0:
                u, v, to_add = best_swap
                selected[u] = False
                selected[v] = True
                if len(to_add) > 0:
                    selected[to_add] = True
                current_count = selected.sum()
                improved = True

            if not improved:
                break

        return selected

    # Strategy 1: Greedy minimum degree
    selected = greedy_min_degree(conflicts, N2, seed=42)
    selected = add_free_vertices(selected, conflicts_int, N2)
    count = selected.sum()
    if count > best_count:
        best_count = count
        best_selected = selected.copy()

    # Strategy 2: Spectral ordering (most negative Fiedler first)
    order = np.argsort(fiedler)
    selected = greedy_ordering(conflicts, N2, order)
    selected = add_free_vertices(selected, conflicts_int, N2)
    count = selected.sum()
    if count > best_count:
        best_count = count
        best_selected = selected.copy()

    # Strategy 3: Spectral ordering (most positive Fiedler first)
    order = np.argsort(-fiedler)
    selected = greedy_ordering(conflicts, N2, order)
    selected = add_free_vertices(selected, conflicts_int, N2)
    count = selected.sum()
    if count > best_count:
        best_count = count
        best_selected = selected.copy()

    # Strategy 4: Kempe-chain improvement on best so far
    if best_selected is not None:
        selected = kempe_chain_improve(best_selected, conflicts, N2, max_iters=5000, time_limit=deadline - 20)
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    # Strategy 5: Multiple random restarts with incremental degree updates
    for restart in range(50):
        if time.time() > deadline - 30:
            break

        seed = 100 + restart
        selected = greedy_min_degree(conflicts, N2, seed=seed)
        selected = add_free_vertices(selected, conflicts_int, N2)
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    # Strategy 6: Kempe-chain improvement on best after restarts
    if best_selected is not None:
        selected = kempe_chain_improve(best_selected, conflicts, N2, max_iters=5000, time_limit=deadline - 10)
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    # Strategy 7b: Additional Kempe passes with different random seeds for tie-breaking
    if best_selected is not None and time.time() < deadline - 5:
        for extra_pass in range(3):
            if time.time() > deadline - 5:
                break
            selected = kempe_chain_improve(best_selected, conflicts, N2, max_iters=2000, time_limit=deadline - 5)
            count = selected.sum()
            if count > best_count:
                best_count = count
                best_selected = selected.copy()
            else:
                break

    # Strategy 7: Final polish - add all free vertices
    if best_selected is not None:
        selected = add_free_vertices(best_selected, conflicts_int, N2)
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
        # One more Kempe pass on the polished set
        if time.time() < deadline - 3:
            selected = kempe_chain_improve(best_selected, conflicts, N2, max_iters=1000, time_limit=deadline - 3)
            count = selected.sum()
            if count > best_count:
                best_count = count
                best_selected = selected.copy()
            selected = add_free_vertices(selected, conflicts_int, N2)
            count = selected.sum()
            if count > best_count:
                best_count = count
                best_selected = selected.copy()

    result_v2 = vectors_v2[best_selected]

    # Add all V1 vectors (they're all compatible with V2 and each other)
    final_vectors = np.vstack([result_v2, vectors_v1])

    # Verify the constraint: min pairwise distance >= max norm
    if len(final_vectors) > 1:
        norms = np.sqrt(np.sum(final_vectors.astype(np.float64)**2, axis=1))
        max_norm = np.max(norms)

        n_final = len(final_vectors)
        valid = np.ones(n_final, dtype=bool)

        batch_size = 500
        for i in range(0, n_final, batch_size):
            if time.time() > deadline - 2:
                break
            batch = final_vectors[i:i+batch_size]
            if len(batch) == 0:
                break
            dots_batch = batch @ final_vectors.T
            norms_sq = np.sum(batch.astype(np.float64)**2, axis=1)
            all_norms_sq = np.sum(final_vectors.astype(np.float64)**2, axis=1)
            dist_sq = norms_sq[:, None] + all_norms_sq[None, :] - 2 * dots_batch

            for r in range(len(batch)):
                idx_in_final = i + r
                if not valid[idx_in_final]:
                    continue
                d_sq = dist_sq[r, :]
                d_sq[idx_in_final] = np.inf
                for c in range(n_final):
                    if c == idx_in_final:
                        continue
                    if not valid[c]:
                        continue
                    if d_sq[c] < max_norm**2 - 1e-6:
                        valid[idx_in_final] = False
                        valid[c] = False
                        break

        final_vectors = final_vectors[valid]

    return final_vectors
