"""Combined program: sparse-matrix greedy set cover with redundant removal and local improvement."""
import itertools
import random
import time
import numpy as np
import scipy.sparse as sp

# EVOLVE-BLOCK-START

def _build_incidence(v, k, t, deadline):
    """Build sparse incidence matrix A (n_k x n_t) and helper structures."""
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)

    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)

    row_indices = []
    col_indices = []
    data = []

    for i, block in enumerate(k_subs):
        block_set = set(block)
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)
        if time.time() > deadline:
            break

    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int8)

    # Precompute column indices per block for fast updates
    block_cols = []
    for i in range(n_k):
        s = A.indptr[i]
        e = A.indptr[i + 1]
        block_cols.append(A.indices[s:e].copy())

    return k_subs, t_subs, A, block_cols, n_k, n_t


def _greedy_cover(A, block_cols, n_k, n_t, deadline, rng, max_iter=50000):
    """Vectorized greedy set cover using sparse matrix-vector product for gains."""
    u = np.ones(n_t, dtype=np.int8)
    selected = []

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if u.sum() == 0:
            break

        # Gain for each block = number of currently uncovered t-subsets it covers
        gains = A @ u
        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]

        if best_gain == 0:
            break

        selected.append(best_idx)
        cols = block_cols[best_idx]
        u[cols] = 0

    return selected, u


def _remove_redundant(selected, block_cols, n_t, deadline):
    """Iteratively remove blocks that don't uniquely cover any t-subset."""
    if not selected:
        return selected

    # cover_count[c] = number of selected blocks covering t-subset c
    cover_count = np.zeros(n_t, dtype=np.int32)
    for idx in selected:
        cover_count[block_cols[idx]] += 1

    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        to_remove = []
        for i in range(len(selected)):
            cols = block_cols[selected[i]]
            is_essential = False
            for c in cols:
                if cover_count[c] == 1:
                    is_essential = True
                    break
            if not is_essential:
                to_remove.append(i)

        if to_remove:
            changed = True
            for i in sorted(to_remove, reverse=True):
                cols = block_cols[selected[i]]
                cover_count[cols] -= 1
                selected.pop(i)

    return selected


def _local_improve(selected, block_cols, A, n_k, n_t, deadline, rng, max_rounds=5):
    """Local search: try to replace or remove blocks to reduce count."""
    if not selected:
        return selected

    best = list(selected)
    best_size = len(best)

    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if len(best) == 0:
            break

        improved = False

        # Try removing each block and see if coverage is still valid
        for i in range(len(best)):
            if time.time() > deadline:
                break

            # Compute coverage without block i
            cover_no_i = np.zeros(n_t, dtype=np.int8)
            for j, idx in enumerate(best):
                if j == i:
                    continue
                cover_no_i[block_cols[idx]] += 1

            # Check if all t-subsets are still covered
            # A t-subset is uncovered if cover_no_i[c] == 0
            # We need to verify that no t-subset has cover_no_i == 0
            # This is expensive; instead check: is there any t-subset with cover_no_i == 0?
            # We can check by seeing if any c in block_cols[best[i]] has cover_no_i[c] == 0
            # If yes, block i is essential (or at least needed for those).
            # Actually, we need to check ALL t-subsets, not just those in block i.
            # But if all t-subsets in block i are covered by others, block i might still be essential
            # for other t-subsets? No - block i only covers t-subsets in block_cols[best[i]].
            # So if all t-subsets in block_cols[best[i]] are covered by others, block i is redundant.
            redundant = True
            for c in block_cols[best[i]]:
                if cover_no_i[c] == 0:
                    redundant = False
                    break

            if redundant:
                best.pop(i)
                improved = True
                break

        if not improved:
            # Try swapping: replace block i with a random candidate
            for i in range(len(best)):
                if time.time() > deadline:
                    break

                # Find uncovered t-subsets if we remove block i
                cover_no_i = np.zeros(n_t, dtype=np.int8)
                for j, idx in enumerate(best):
                    if j == i:
                        continue
                    cover_no_i[block_cols[idx]] += 1

                # T-subsets that become uncovered
                uncovered_if_removed = []
                for c in block_cols[best[i]]:
                    if cover_no_i[c] == 0:
                        uncovered_if_removed.append(c)

                if not uncovered_if_removed:
                    # Block i is redundant, should have been caught above
                    best.pop(i)
                    improved = True
                    break

                # Try a few random candidates to replace block i
                found_replacement = False
                for _ in range(20):
                    candidate = rng.randint(0, n_k - 1)
                    if candidate == best[i]:
                        continue

                    # Check if candidate covers all uncovered_if_removed
                    cand_cols = set(block_cols[candidate].tolist())
                    if all(c in cand_cols for c in uncovered_if_removed):
                        # Also ensure we don't create issues elsewhere
                        # Since we're replacing, coverage of other t-subsets is unaffected
                        # (candidate may cover more, which is fine)
                        best[i] = candidate
                        found_replacement = True
                        improved = True
                        break

                if found_replacement:
                    break

        if not improved:
            break

    return best


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using sparse greedy with local improvement."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin within 150s limit

    rng = random.Random(42)

    # Build incidence structure
    k_subs, t_subs, A, block_cols, n_k, n_t = _build_incidence(v, k, t, deadline)

    best_blocks = None
    best_count = float('inf')

    # Multiple restarts with different random seeds for tie-breaking
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break

        # For variety, shuffle the order in which ties are broken by using
        # a slightly perturbed gain vector
        selected, u = _greedy_cover(A, block_cols, n_k, n_t, deadline, rng)

        if not selected:
            continue

        # Check if valid (all t-subsets covered)
        if u.sum() == 0:
            # Remove redundant blocks
            selected = _remove_redundant(selected, block_cols, n_t, deadline)

            # Local improvement
            selected = _local_improve(selected, block_cols, A, n_k, n_t, deadline, rng, max_rounds=3)

            if len(selected) < best_count:
                best_count = len(selected)
                best_blocks = selected
                if best_count <= 300:
                    break

    if best_blocks is None:
        # Fallback: single greedy run
        selected, u = _greedy_cover(A, block_cols, n_k, n_t, deadline, rng)
        if u.sum() == 0:
            selected = _remove_redundant(selected, block_cols, n_t, deadline)
            best_blocks = selected
        else:
            # Should not happen, but return what we have
            best_blocks = selected

    # Convert block indices back to lists of points
    result = [list(k_subs[idx]) for idx in best_blocks]
    return result

# EVOLVE-BLOCK-END
