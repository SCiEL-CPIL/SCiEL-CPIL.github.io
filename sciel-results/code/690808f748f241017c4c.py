"""Improved program: expanded candidate pool + max-min greedy for kissing number in dimension 11."""
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    D = 11

    # Generate weight-4 vectors (norm=2): C(11,4)*16 = 5280
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Generate weight-2 vectors (norm=sqrt(2)): C(11,2)*4 = 220
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5500

    # Precompute norms squared for each candidate
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)  # 4 for weight-4, 2 for weight-2
    max_norm_sq = norms_sq.max()  # 4

    # Precompute the full dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*(v_i·v_j) >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    norm_i = norms_sq[:, np.newaxis]  # (n, 1)
    norm_j = norms_sq[np.newaxis, :]  # (1, n)
    threshold = (norm_i + norm_j - max_norm_sq) / 2.0  # (n, n)
    # compat[i,j] = True if dot[i,j] <= threshold[i,j]
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # Max-min greedy: at each step, pick the candidate that is compatible with the most
    # remaining candidates (maximin / max-min heuristic)
    def greedy_maxmin(time_limit):
        """Max-min greedy: repeatedly pick the vertex with max degree in the remaining graph."""
        selected = []
        # remaining: boolean mask of which candidates are still available
        remaining = np.ones(n, dtype=np.uint8)
        # deg: for each remaining candidate, how many other remaining candidates it's compatible with
        # We'll maintain a running count

        while time.time() < time_limit:
            if not remaining.any():
                break
            # Compute degree of each remaining candidate within the remaining set
            # deg[i] = sum of compat[i, j] for j where remaining[j] == 1 and j != i
            # Vectorized: for each i, dot compat[i] with remaining
            # But we need to exclude self. Since compat[i,i]=1, we subtract 1 if remaining[i].
            deg = (compat * remaining[np.newaxis, :]).sum(axis=1).astype(np.int32)
            deg[remaining == 0] = -1  # exclude non-remaining
            # Among remaining, pick the one with max degree
            max_deg = deg.max()
            if max_deg < 0:
                break
            # Pick the first candidate with max degree among remaining
            candidates_with_max = np.where((deg == max_deg) & (remaining > 0))[0]
            if len(candidates_with_max) == 0:
                break
            new_idx = candidates_with_max[0]
            selected.append(new_idx)
            # Update remaining: must be compatible with new_idx
            remaining = remaining & compat[new_idx]
            remaining[new_idx] = 0

        return selected

    def local_search_improve(selected_indices, time_limit, max_iter=100):
        """Try to improve the selected set by adding vectors where possible."""
        selected = list(selected_indices)
        if not selected:
            return selected

        # Build a compatibility mask: mask[j]=1 if j is compatible with all selected
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        for _ in range(max_iter):
            if time.time() > time_limit:
                break
            candidates_to_add = np.where(compat_with_all > 0)[0]
            if len(candidates_to_add) == 0:
                break
            # Pick the one with max degree among candidates_to_add
            sub_deg = (compat[candidates_to_add] * candidates_to_add[np.newaxis == False] if False else 1).sum(axis=1)
            # Simpler: just pick the one with highest degree in the full graph among candidates_to_add
            degrees = (compat * (compat_with_all > 0)[np.newaxis, :]).sum(axis=1)
            best_local = candidates_to_add[np.argmax(degrees[candidates_to_add])]
            selected.append(best_local)
            compat_with_all = compat_with_all & compat[best_local]
            compat_with_all[best_local] = 0

        return selected

    # Strategy: multiple random restarts with max-min greedy
    best_selected = []
    np.random.seed(42)

    # First run: pure max-min greedy
    if time.time() < deadline - 20.0:
        sel = greedy_maxmin(deadline - 20.0)
        if len(sel) > len(best_selected):
            best_selected = sel

    # More random restarts: perturb the candidate ordering by random permutation of indices
    # For max-min, we can randomize ties by permuting candidates
    for trial in range(80):
        if time.time() > deadline - 15.0:
            break
        # Create a perturbed compatibility matrix by adding tiny random noise to break ties
        # Or simpler: random subset of candidates to start with, then max-min
        # Alternative: just run max-min but with random tie-breaking via random permutation
        perm = np.random.permutation(n)
        # We need to modify greedy to use permuted order for tie-breaking
        # Simpler approach: randomly remove some candidates first, then max-min on the rest
        # Let's just do: random subset of size n//2 + n//4, then max-min
        subset_size = n // 2 + np.random.randint(0, n // 4)
        subset_indices = np.random.choice(n, size=min(subset_size, n), replace=False)
        # Run max-min restricted to subset
        remaining = np.zeros(n, dtype=np.uint8)
        remaining[subset_indices] = 1
        selected = []
        time_limit = deadline - 15.0
        while time.time() < time_limit:
            if not remaining.any():
                break
            deg = (compat * remaining[np.newaxis, :]).sum(axis=1).astype(np.int32)
            deg[remaining == 0] = -1
            max_deg = deg.max()
            if max_deg < 0:
                break
            candidates_with_max = np.where((deg == max_deg) & (remaining > 0))[0]
            if len(candidates_with_max) == 0:
                break
            new_idx = candidates_with_max[0]
            selected.append(new_idx)
            remaining = remaining & compat[new_idx]
            remaining[new_idx] = 0

        if len(selected) > len(best_selected):
            best_selected = selected

    # Local search on the best
    if best_selected and time.time() < deadline - 5.0:
        improved = local_search_improve(best_selected, deadline - 5.0, max_iter=50)
        if len(improved) > len(best_selected):
            best_selected = improved

    if not best_selected:
        # Fallback
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_selected, dtype=np.int64)]
    return result.astype(np.int64)
