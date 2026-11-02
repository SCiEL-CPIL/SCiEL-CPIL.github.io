"""Improved program: spectral relaxation + min-degree greedy + swap local search for kissing number in dimension 11."""
import numpy as np
import itertools
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s limit

    D = 11

    # Generate weight-4 vectors (norm^2=4): C(11,4)*16 = 5280
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Generate weight-2 vectors (norm^2=2): C(11,2)*4 = 220
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Generate weight-1 vectors (norm^2=1): 22
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5522

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected (i != j)
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    def greedy_min_degree(remaining_init, time_limit):
        """Min-degree greedy: pick vertex with fewest conflicts among remaining."""
        remaining = remaining_init.copy()
        selected = []
        while time.time() < time_limit and remaining.any():
            # degree[i] = number of conflicts with other remaining vertices
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1  # high value so they won't be picked
            min_deg = deg.min()
            if min_deg >= n:
                break
            # Pick first candidate with min degree among remaining
            mask = (deg == min_deg) & (remaining > 0)
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            selected.append(new_idx)
            # Remove new_idx and all its conflicts from remaining
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0
        return selected

    def swap_local_search(selected_list, time_limit, max_rounds=200):
        """Swap-based local search: try to grow by swapping one selected vector for two."""
        selected = list(selected_list)
        sel_set = set(selected)
        # compat_with_all[j] = 1 if j is compatible with all currently selected
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        for _ in range(max_rounds):
            if time.time() > time_limit:
                break
            # Try simple addition first
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                # Pick the one with min degree in full graph (least constraining)
                full_deg = conflict.sum(axis=1)
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                continue

            # Try swap: remove one selected, add two new ones
            # For each selected s, find candidates compatible with (selected - {s})
            # that are NOT compatible with s (i.e., they conflict with s)
            # We need at least 2 such candidates that are mutually compatible
            found_swap = False
            # Limit: only try swaps for a random subset of selected to save time
            if len(selected) > 30:
                trial_indices = np.random.choice(len(selected), size=min(10, len(selected)), replace=False)
            else:
                trial_indices = range(len(selected))

            for si in trial_indices:
                if time.time() > time_limit:
                    break
                s = selected[si]
                # Candidates compatible with all selected except s, and NOT s itself
                # = compat_with_all | (conflict with s is OK)
                # A candidate j is compatible with (selected - {s}) iff
                #   compat[j, s'] = 1 for all s' in selected, s' != s
                # This is equivalent to: compat_with_all[j] | (j is compatible with all except s)
                # Simpler: recompute = product of compat[s'] for s' != s
                # But that's expensive. Instead:
                # j is compatible with all except s iff:
                #   for all s' in selected: s' != s => compat[j,s']=1
                #   AND compat[j,s] can be 0 or 1 (we don't care about s)
                # The candidates that are compatible with all selected EXCEPT possibly s:
                #   = (compat_with_all) OR (compatible with all selected except s)
                # Let's compute: compatible_with_except_s[j] = 1 if j compatible with all s' in selected, s' != s
                # = product over s' in selected, s'!=s of compat[j, s']
                # = compat_with_all[j] * (1 if compat[j,s]=1 else ...) 
                # Actually: compat_with_all[j] = prod_{s' in selected} compat[j,s']
                # compatible_with_except_s[j] = compat_with_all[j] / compat[j,s] (if compat[j,s] is 0 or 1)
                # If compat[j,s] = 1: compatible_with_except_s[j] = compat_with_all[j]
                # If compat[j,s] = 0: compatible_with_except_s[j] = 0 (since compat_with_all[j] = 0)
                # Wait, that's not right. compat_with_all[j] = 0 if j conflicts with ANY selected.
                # If j conflicts only with s, then compat_with_all[j] = 0 but compatible_with_except_s[j] = 1.
                # So we need a different computation.
                
                # Compute: for j, is j compatible with all selected except s?
                # = AND over s' in selected, s' != s: compat[j, s']
                # Let's compute this vectorized:
                # mask_except_s: (n,) = 1 if j compatible with all s' in selected, s' != s
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[:, s2]
                mask_except_s[s] = 0  # exclude s itself
                mask_except_s[list(sel_set)] = 0  # exclude all selected (we want NEW candidates)
                
                candidates_for_swap = np.where(mask_except_s > 0)[0]
                if len(candidates_for_swap) < 2:
                    continue
                
                # Among these, find two that are mutually compatible
                # Pick the one with min full degree first, then check if any other is compatible
                full_deg = conflict.sum(axis=1)
                sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]
                
                found_pair = False
                for i in range(min(20, len(sorted_cands))):
                    c1 = sorted_cands[i]
                    # Find any c2 in sorted_cands that is compatible with c1
                    compat_with_c1 = compat[c1]
                    for j in range(i + 1, min(40, len(sorted_cands))):
                        c2 = sorted_cands[j]
                        if compat_with_c1[c2]:
                            # Do the swap: remove s, add c1 and c2
                            selected.remove(s)
                            sel_set.discard(s)
                            selected.append(c1)
                            sel_set.add(c1)
                            selected.append(c2)
                            sel_set.add(c2)
                            # Recompute compat_with_all
                            compat_with_all = np.ones(n, dtype=np.uint8)
                            for s3 in selected:
                                compat_with_all = compat_with_all & compat[s3]
                            for s3 in selected:
                                compat_with_all[s3] = 0
                            found_swap = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_swap:
                    break

            if not found_swap:
                break

        return selected

    # Strategy: spectral relaxation to find a large independent set, then greedy refinement
    best_selected = []
    np.random.seed(123)

    # Build sparse conflict matrix for spectral methods
    conflict_sparse = sp.csr_matrix(conflict)

    # Spectral relaxation: compute leading eigenvector of graph Laplacian
    # L = D - A where A is adjacency (conflict) matrix
    degrees = np.array(conflict_sparse.sum(axis=1)).flatten()
    D_mat = sp.diags(degrees)
    L = D_mat - conflict_sparse  # Graph Laplacian

    # Find the eigenvector corresponding to the smallest non-zero eigenvalue
    # This gives a soft assignment that approximates a large independent set
    try:
        # Use shift-invert to find small eigenvalues efficiently
        # We want eigenvalues near 0 (but not exactly 0 which corresponds to constant vector)
        eigvals, eigvecs = spla.eigsh(L, k=10, sigma=0.0, which='LM', v0=np.ones(n)/np.sqrt(n))
        # Sort by eigenvalue
        idx_sort = np.argsort(eigvals)
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]
        
        # Use the first non-trivial eigenvector (skip the trivial constant one)
        # The trivial eigenvector has eigenvalue 0
        # Find the first eigenvector with eigenvalue > 1e-6
        spectral_vec = None
        for i in range(len(eigvals)):
            if eigvals[i] > 1e-6:
                spectral_vec = eigvecs[:, i]
                break
        
        if spectral_vec is None:
            spectral_vec = eigvecs[:, 0]
    except Exception:
        spectral_vec = None

    # If spectral method failed, use degree-based heuristic as fallback
    if spectral_vec is None:
        # Use negative degree as a proxy (low-degree nodes are less constraining)
        spectral_vec = -np.array(conflict.sum(axis=1), dtype=np.float64)

    # Normalize spectral vector
    spectral_vec = spectral_vec - spectral_vec.mean()
    spectral_vec = spectral_vec / (np.linalg.norm(spectral_vec) + 1e-10)

    # Greedy selection based on spectral scores: pick nodes in order of highest score,
    # but only if compatible with all previously selected
    spectral_order = np.argsort(-spectral_vec)  # Descending order of spectral score
    selected_spectral = []
    selected_set = set()
    for idx in spectral_order:
        if time.time() > deadline - 80.0:
            break
        if idx in selected_set:
            continue
        # Check compatibility with all selected
        compatible = True
        for s in selected_spectral:
            if not compat[idx, s]:
                compatible = False
                break
        if compatible:
            selected_spectral.append(idx)
            selected_set.add(idx)

    if len(selected_spectral) > len(best_selected):
        best_selected = selected_spectral

    # Refine the spectral solution with min-degree greedy on remaining
    if best_selected and time.time() < deadline - 50.0:
        remaining_init = np.ones(n, dtype=np.uint8)
        for s in best_selected:
            remaining_init = remaining_init & (1 - conflict[s])
            remaining_init[s] = 0
        # Continue with min-degree greedy on remaining
        remaining = remaining_init
        extended = list(best_selected)
        while time.time() < deadline - 50.0 and remaining.any():
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & (remaining > 0)
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            extended.append(new_idx)
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0
        if len(extended) > len(best_selected):
            best_selected = extended

    # Also run pure min-degree greedy as baseline
    sel = greedy_min_degree(np.ones(n, dtype=np.uint8), deadline - 40.0)
    if len(sel) > len(best_selected):
        best_selected = sel

    # Random restarts: random subsets
    for trial in range(30):
        if time.time() > deadline - 30.0:
            break
        # Random subset of varying sizes
        frac = np.random.uniform(0.3, 1.0)
        subset_size = int(n * frac)
        subset_indices = np.random.choice(n, size=subset_size, replace=False)
        remaining_init = np.zeros(n, dtype=np.uint8)
        remaining_init[subset_indices] = 1
        sel = greedy_min_degree(remaining_init, deadline - 30.0)
        if len(sel) > len(best_selected):
            best_selected = sel

    # Swap local search on the best
    if best_selected and time.time() < deadline - 10.0:
        improved = swap_local_search(best_selected, deadline - 10.0, max_rounds=300)
        if len(improved) > len(best_selected):
            best_selected = improved

    # One more round of greedy filling on the improved set
    if best_selected and time.time() < deadline - 5.0:
        sel_set = set(best_selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in best_selected:
            compat_with_all = compat_with_all & compat[s]
        for s in best_selected:
            compat_with_all[s] = 0
        addable = np.where(compat_with_all > 0)[0]
        full_deg = conflict.sum(axis=1)
        for idx in addable[np.argsort(full_deg[addable])]:
            if time.time() > deadline - 5.0:
                break
            best_selected.append(idx)
            sel_set.add(idx)
            compat_with_all = compat_with_all & compat[idx]
            compat_with_all[idx] = 0

    if not best_selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_selected, dtype=np.int64)]
    return result.astype(np.int64)
