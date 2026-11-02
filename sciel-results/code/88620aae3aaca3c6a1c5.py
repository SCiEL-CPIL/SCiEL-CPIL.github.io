"""Combined LP-guided greedy with column generation for C(19,7,4) covering."""
import itertools
import time
import random
import heapq
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Build sparse incidence matrix: A[i, j] = 1 if t_subsets[i] ⊆ k_subsets[j]
    # Also build helper lists
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    rows = []
    cols = []
    data = []

    for j, B in enumerate(all_k_subsets):
        for S in itertools.combinations(B, t):
            i = t_sub_to_idx[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)

    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))

    # Solve LP relaxation to get dual variables (weights for t-subsets)
    duals = None
    try:
        c = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds = [(0, 1)] * n_k
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # For minimization with <= constraints, marginals are <= 0
            # The dual variable y_i for constraint >= 1 is -marginal[i]
            duals = -res.ineqlin.marginals
            duals = np.maximum(duals, 0)
            # Normalize to sum to 1 for stability
            if duals.sum() > 0:
                duals = duals / duals.sum()
    except Exception:
        duals = None

    if duals is None:
        # Fallback: uniform weights
        duals = np.ones(n_t) / n_t

    # Precompute block weights from duals
    block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            block_weights[j] = np.sum(duals[subsets_of_block[j]])

    best_blocks = None
    best_count = float('inf')

    # Multiple strategies: LP-guided greedy, standard greedy, column-gen hybrid
    n_restarts = 40

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(42 + restart * 104729)

        covered = [False] * n_t
        covered_count = 0
        blocks = []

        # Choose strategy based on restart
        strategy = restart % 3  # 0: LP-guided, 1: standard greedy, 2: column-gen

        if strategy == 0:
            # LP-guided greedy with heap
            blocks = _lp_guided_greedy(v, k, t, all_k_subsets, subsets_of_block,
                                        blocks_of_subset, block_weights, duals,
                                        t_sub_to_idx, deadline)
            if not time_ok():
                break
        elif strategy == 1:
            # Standard max-coverage greedy with randomization
            blocks = _standard_greedy(v, k, t, all_k_subsets, subsets_of_block,
                                       blocks_of_subset, t_sub_to_idx, deadline,
                                       random_state=restart)
            if not time_ok():
                break
        else:
            # Column generation approach
            blocks = _column_generation(v, k, t, all_k_subsets, subsets_of_block,
                                         blocks_of_subset, t_sub_to_idx, deadline,
                                         random_state=restart)
            if not time_ok():
                break

        if blocks is None:
            continue

        # Verify coverage
        for i, sub in enumerate(all_t_subsets):
            if not covered[i]:
                # Check if actually covered by the blocks
                pass

        # Re-verify coverage properly
        covered_check = [False] * n_t
        for b in blocks:
            for sub in itertools.combinations(b, t):
                idx = t_sub_to_idx[sub]
                covered_check[idx] = True

        if all(covered_check):
            # Deletion phase: iterative removal of redundant blocks
            if time_ok():
                blocks = _deletion_phase(blocks, v, k, t, t_sub_to_idx, deadline)

            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)

            # Early stop if we've done very well
            if best_count <= 145:
                break
        else:
            # Not fully covered - try to fix with additional blocks
            if time_ok():
                blocks = _fix_coverage(blocks, v, k, t, all_k_subsets,
                                        subsets_of_block, blocks_of_subset,
                                        t_sub_to_idx, deadline)
                if all(covered_check):  # re-check
                    if time_ok():
                        blocks = _deletion_phase(blocks, v, k, t, t_sub_to_idx, deadline)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)

    if best_blocks is None:
        # Final fallback: pure greedy
        blocks = _standard_greedy(v, k, t, all_k_subsets, subsets_of_block,
                                   blocks_of_subset, t_sub_to_idx, deadline,
                                   random_state=999)
        if blocks and all(_is_covered(blocks, v, t, t_sub_to_idx)):
            return [list(b) for b in blocks]
        return []

    return [list(b) for b in best_blocks]


def _is_covered(blocks, v, t, t_sub_to_idx):
    """Check if all t-subsets are covered."""
    covered = [False] * len(t_sub_to_idx)
    for b in blocks:
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            covered[idx] = True
    return all(covered)


def _lp_guided_greedy(v, k, t, all_k_subsets, subsets_of_block, blocks_of_subset,
                       block_weights, duals, t_sub_to_idx, deadline):
    """LP-guided greedy using heap for efficient block selection."""
    n_t = len(t_sub_to_idx)
    n_k = len(all_k_subsets)

    # Work on copies
    cur_weights = block_weights.copy()
    covered = [False] * n_t
    picked = [False] * n_k
    blocks = []

    # Initialize heap with all blocks
    heap = [(-cur_weights[j], j) for j in range(n_k)]
    heapq.heapify(heap)

    while time.time() < deadline:
        if not heap:
            break

        neg_w, j = heapq.heappop(heap)
        w = -neg_w

        if picked[j]:
            continue
        # Lazy deletion: check if weight is stale
        if w > cur_weights[j] + 1e-9:
            continue
        if w <= 1e-9:
            continue

        # Check if this block covers any uncovered subsets
        has_new = False
        for s_idx in subsets_of_block[j]:
            if not covered[s_idx]:
                has_new = True
                break
        if not has_new:
            continue

        # Select block j
        picked[j] = True
        blocks.append(list(all_k_subsets[j]))

        # Update weights for blocks sharing subsets with j
        for s_idx in subsets_of_block[j]:
            if not covered[s_idx]:
                covered[s_idx] = True
                y_s = duals[s_idx]
                if y_s > 0:
                    for b_prime in blocks_of_subset[s_idx]:
                        if not picked[b_prime]:
                            cur_weights[b_prime] -= y_s
                            heapq.heappush(heap, (-cur_weights[b_prime], b_prime))

        # Check if all covered
        if all(covered):
            break

    # Verify and fix if needed
    if not all(covered):
        remaining = [i for i in range(n_t) if not covered[i]]
        if remaining:
            # Add blocks to cover remaining
            for idx in remaining:
                # Find a block containing this t-subset
                for b_idx in blocks_of_subset[idx]:
                    if not picked[b_idx]:
                        picked[b_idx] = True
                        blocks.append(list(all_k_subsets[b_idx]))
                        for s_idx in subsets_of_block[b_idx]:
                            covered[s_idx] = True
                        break

    return blocks if all(covered) else None


def _standard_greedy(v, k, t, all_k_subsets, subsets_of_block, blocks_of_subset,
                      t_sub_to_idx, deadline, random_state=0):
    """Standard max-coverage greedy with some randomization."""
    rng = random.Random(random_state)
    n_t = len(t_sub_to_idx)
    n_k = len(all_k_subsets)

    covered = [False] * n_t
    covered_count = 0
    blocks = []
    picked = [False] * n_k

    while time.time() < deadline:
        if covered_count >= n_t:
            break

        # Find best block by max new coverage
        best_j = -1
        best_gain = 0

        # Sample a subset of blocks if too many to check all
        if n_k <= 2000:
            candidates = range(n_k)
        else:
            candidates = rng.sample(range(n_k), 2000)

        for j in candidates:
            if picked[j]:
                continue
            gain = 0
            for s_idx in subsets_of_block[j]:
                if not covered[s_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j < 0 or best_gain == 0:
            # Try all blocks if sampling failed
            for j in range(n_k):
                if picked[j]:
                    continue
                gain = 0
                for s_idx in subsets_of_block[j]:
                    if not covered[s_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_j = j

        if best_j < 0 or best_gain == 0:
            break

        picked[best_j] = True
        blocks.append(list(all_k_subsets[best_j]))
        for s_idx in subsets_of_block[best_j]:
            if not covered[s_idx]:
                covered[s_idx] = True
                covered_count += 1

    if not all(covered):
        # Fix remaining
        for idx in range(n_t):
            if not covered[idx]:
                for b_idx in blocks_of_subset[idx]:
                    if not picked[b_idx]:
                        picked[b_idx] = True
                        blocks.append(list(all_k_subsets[b_idx]))
                        for s_idx in subsets_of_block[b_idx]:
                            covered[s_idx] = True
                        break

    return blocks if all(covered) else None


def _column_generation(v, k, t, all_k_subsets, subsets_of_block, blocks_of_subset,
                        t_sub_to_idx, deadline, random_state=0):
    """Column generation: iteratively add most valuable blocks."""
    rng = random.Random(random_state)
    n_t = len(t_sub_to_idx)
    n_k = len(all_k_subsets)

    covered = [False] * n_t
    covered_count = 0
    blocks = []
    picked = [False] * n_k

    # Start with a few random blocks
    initial = min(10, n_k)
    init_indices = rng.sample(range(n_k), initial)
    for j in init_indices:
        picked[j] = True
        blocks.append(list(all_k_subsets[j]))
        for s_idx in subsets_of_block[j]:
            if not covered[s_idx]:
                covered[s_idx] = True
                covered_count += 1

    # Iteratively add blocks
    max_iter = 500
    for _ in range(max_iter):
        if time.time() >= deadline:
            break
        if covered_count >= n_t:
            break

        # Find uncovered t-subsets
        uncovered = [i for i in range(n_t) if not covered[i]]
        if not uncovered:
            break

        # For each uncovered t-subset, find blocks that cover it
        # Score blocks by number of uncovered subsets they cover
        block_scores = {}
        for idx in uncovered:
            for b_idx in blocks_of_subset[idx]:
                if not picked[b_idx]:
                    block_scores[b_idx] = block_scores.get(b_idx, 0) + 1

        if not block_scores:
            break

        # Pick the highest-scoring block
        best_j = max(block_scores, key=lambda x: (block_scores[x], rng.random()))
        picked[best_j] = True
        blocks.append(list(all_k_subsets[best_j]))

        # Update coverage
        for s_idx in subsets_of_block[best_j]:
            if not covered[s_idx]:
                covered[s_idx] = True
                covered_count += 1

    if not all(covered):
        # Fix remaining
        for idx in range(n_t):
            if not covered[idx]:
                for b_idx in blocks_of_subset[idx]:
                    if not picked[b_idx]:
                        picked[b_idx] = True
                        blocks.append(list(all_k_subsets[b_idx]))
                        for s_idx in subsets_of_block[b_idx]:
                            covered[s_idx] = True
                        break

    return blocks if all(covered) else None


def _fix_coverage(blocks, v, k, t, all_k_subsets, subsets_of_block, blocks_of_subset,
                   t_sub_to_idx, deadline):
    """Add blocks to fix any uncovered t-subsets."""
    n_t = len(t_sub_to_idx)
    covered = [False] * n_t
    for b in blocks:
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            covered[idx] = True

    if all(covered):
        return blocks

    blocks = list(blocks)
    picked_set = set(tuple(b) for b in blocks)

    for idx in range(n_t):
        if not covered[idx]:
            for b_idx in blocks_of_subset[idx]:
                b_tuple = tuple(all_k_subsets[b_idx])
                if b_tuple not in picked_set:
                    picked_set.add(b_tuple)
                    blocks.append(list(all_k_subsets[b_idx]))
                    for s_idx in subsets_of_block[b_idx]:
                        covered[s_idx] = True
                    break

    return blocks


def _deletion_phase(blocks, v, k, t, t_sub_to_idx, deadline):
    """Iteratively remove redundant blocks."""
    n_t = len(t_sub_to_idx)
    blocks = list(blocks)

    # Count how many blocks cover each t-subset
    tsub_count = [0] * n_t
    for b in blocks:
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            tsub_count[idx] += 1

    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find a removable block
        for i in range(len(blocks)):
            b = blocks[i]
            removable = True
            for sub in itertools.combinations(b, t):
                idx = t_sub_to_idx[sub]
                if tsub_count[idx] <= 1:
                    removable = False
                    break
            if removable:
                # Remove block i
                for sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[sub]
                    tsub_count[idx] -= 1
                blocks.pop(i)
                changed = True
                break

    return blocks
