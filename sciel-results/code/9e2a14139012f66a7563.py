"""Combined covering-design program: vectorized greedy + inverted-index gains + local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

def _precompute(v, k, t):
    """Precompute t-subset indices, k-block list, block->t-index matrix, and inverted index."""
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    local_t_combos = list(itertools.combinations(range(k), t))
    n_t_in_k = len(local_t_combos)

    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    for j, lc in enumerate(local_t_combos):
        lc_list = list(lc)
        for i, block in enumerate(k_subsets):
            t_combo = tuple(block[p] for p in lc_list)
            block_t_indices[i, j] = t_sub_to_idx[t_combo]

    # Inverted index: for each t-subset, the list of blocks containing it
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_blocks[t_idx].append(i)

    # Coverage count (how many blocks cover each t-subset) - static, over all blocks
    static_coverage = np.zeros(n_t, dtype=np.int32)
    for i in range(n_k):
        for j in range(n_t_in_k):
            static_coverage[block_t_indices[i, j]] += 1

    return t_subsets, t_sub_to_idx, n_t, k_subsets, n_k, block_t_indices, n_t_in_k, t_to_blocks, static_coverage


def _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, static_coverage,
                  start_time, deadline, seed=0, pivot_strategy="constrained", noise=0.0):
    """Greedy set cover with incremental gain updates and pivot selection."""
    rng = random.Random(seed)

    uncovered = np.ones(n_t, dtype=bool)
    uncovered_count = n_t
    selected_indices = []

    # gains[i] = number of uncovered t-subsets in block i
    gains = np.full(n_k, n_t_in_k, dtype=np.int32)

    while uncovered_count > 0:
        if time.time() - start_time > deadline - 5.0:
            return None

        # Choose pivot t-set
        if pivot_strategy == "constrained":
            # Pick uncovered t-set with fewest covering blocks (most constrained)
            # Sample candidates for speed
            uncovered_idx = np.where(uncovered)[0]
            if len(uncovered_idx) <= 30:
                candidates = uncovered_idx.tolist()
            else:
                candidates = rng.sample(uncovered_idx.tolist(), 30)
            best_pivot = -1
            best_cnt = float('inf')
            for ts_idx in candidates:
                cnt = static_coverage[ts_idx]
                if cnt < best_cnt:
                    best_cnt = cnt
                    best_pivot = ts_idx
            if best_pivot == -1:
                best_pivot = uncovered_idx[0]
        else:
            uncovered_idx = np.where(uncovered)[0]
            best_pivot = rng.choice(uncovered_idx.tolist())

        # Among blocks covering the pivot, pick the one with max gain
        candidates = t_to_blocks[best_pivot]
        if not candidates:
            break

        best_idx = -1
        best_gain = -1
        for b in candidates:
            g = gains[b]
            if noise > 0:
                g = g + rng.uniform(0, noise)
            if g > best_gain:
                best_gain = g
                best_idx = b

        if best_idx == -1 or best_gain <= 0:
            break

        selected_indices.append(best_idx)

        # Update uncovered and gains incrementally
        block_ts = block_t_indices[best_idx]
        for t_idx in block_ts:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                uncovered_count -= 1
                for b in t_to_blocks[t_idx]:
                    gains[b] -= 1

        if uncovered_count <= 0:
            break

    return selected_indices


def _remove_redundant(selected_indices, block_t_indices, n_t, start_time, deadline):
    """Remove blocks whose t-subsets are all covered by other selected blocks."""
    if not selected_indices:
        return selected_indices

    coverage_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected_indices:
        coverage_count[block_t_indices[bi]] += 1

    remaining = list(selected_indices)
    changed = True
    while changed:
        if time.time() - start_time > deadline - 3.0:
            break
        changed = False
        for i in range(len(remaining)):
            bi = remaining[i]
            ts = block_t_indices[bi]
            if np.all(coverage_count[ts] > 1):
                coverage_count[ts] -= 1
                remaining.pop(i)
                changed = True
                break
    return remaining


def _local_swap(selected_indices, block_t_indices, n_t, n_k, t_to_blocks, start_time, deadline, seed=0):
    """Try to swap a block for another that covers more currently-scarce t-sets."""
    if not selected_indices:
        return selected_indices

    rng = random.Random(seed)
    coverage_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected_indices:
        coverage_count[block_t_indices[bi]] += 1

    current = list(selected_indices)
    improved = True
    while improved:
        if time.time() - start_time > deadline - 3.0:
            break
        improved = False
        # Try removing each block and replacing with a better one
        for i in range(len(current)):
            if time.time() - start_time > deadline - 3.0:
                return current
            bi = current[i]
            ts = block_t_indices[bi]
            # t-subsets that would become uncovered if we remove bi
            scarce = ts[coverage_count[ts] == 1]
            if len(scarce) == 0:
                # bi is redundant; just remove it
                coverage_count[ts] -= 1
                current.pop(i)
                improved = True
                break
            # Find a block that covers all scarce t-subsets and adds net benefit
            # Candidates: blocks containing the first scarce t-set
            first_scarce = scarce[0]
            candidates = t_to_blocks[first_scarce]
            best_new = -1
            best_score = -1
            for nb in candidates:
                if nb == bi:
                    continue
                nts = block_t_indices[nb]
                # Must cover all scarce
                if not np.all(coverage_count[nts] == 0) or False:
                    pass
                # Check coverage of all scarce
                covers_all = True
                for s in scarce:
                    if s not in nts:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                # Score: number of currently-uncovered-by-bi t-sets that nb covers
                # i.e., t-sets in nb that are not in bi
                # Approximate: count of t-sets in nb with coverage_count == 0 (would be newly covered)
                # But coverage_count includes bi, so for t-sets in nb but not bi,
                # coverage_count >= 1 means covered by others.
                # Net benefit = (newly covered) - (lose bi's unique coverage)
                # Since nb covers all scarce, we don't lose scarce.
                # Newly covered = t-sets in nb not in bi that have coverage_count == 1 (only bi covers them? no)
                # Actually if a t-set is in nb but not in bi, and coverage_count == 1, it's covered only by bi? No.
                # coverage_count counts all selected blocks. If t-set in nb not in bi, and coverage_count==1,
                # it means only one selected block covers it, which is not bi (since bi doesn't cover it),
                # so it's covered by that other block. Removing bi doesn't affect it.
                # So net benefit = 0 for those. The real benefit is if nb covers t-sets that no other
                # selected block covers (coverage_count would be 0 if we hadn't selected nb).
                # This is getting complex; use a simpler heuristic:
                # score = number of t-sets in nb that are not in bi (potential new coverage)
                score = len(set(nts.tolist()) - set(ts.tolist()))
                if score > best_score:
                    best_score = score
                    best_new = nb
            if best_new != -1:
                # Perform swap
                coverage_count[ts] -= 1
                nts = block_t_indices[best_new]
                coverage_count[nts] += 1
                current[i] = best_new
                improved = True
                break
    return current


def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin under 120s

    # Precompute
    (t_subsets, t_sub_to_idx, n_t, k_subsets, n_k,
     block_t_indices, n_t_in_k, t_to_blocks, static_coverage) = _precompute(v, k, t)

    best_solution = None
    best_size = float('inf')

    # Strategy 1: Constrained-pivot greedy, deterministic seed
    sol = _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, static_coverage,
                        start_time, deadline, seed=42, pivot_strategy="constrained", noise=0.0)
    if sol:
        sol = _remove_redundant(sol, block_t_indices, n_t, start_time, deadline)
        best_size = len(sol)
        best_solution = sol

    # Strategy 2: Randomized greedy with different seeds
    if time.time() - start_time < deadline - 20.0:
        for seed in range(1, 10):
            if time.time() - start_time > deadline - 10.0:
                break
            sol = _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, static_coverage,
                                start_time, deadline, seed=seed + 100, pivot_strategy="random", noise=0.0)
            if sol:
                sol = _remove_redundant(sol, block_t_indices, n_t, start_time, deadline)
                if len(sol) < best_size:
                    best_size = len(sol)
                    best_solution = sol

    # Strategy 3: Constrained-pivot with noise
    if time.time() - start_time < deadline - 20.0:
        for seed in range(1, 6):
            if time.time() - start_time > deadline - 10.0:
                break
            sol = _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, static_coverage,
                                start_time, deadline, seed=seed + 200, pivot_strategy="constrained", noise=0.1)
            if sol:
                sol = _remove_redundant(sol, block_t_indices, n_t, start_time, deadline)
                if len(sol) < best_size:
                    best_size = len(sol)
                    best_solution = sol

    # Local swap improvement on best solution
    if best_solution and time.time() - start_time < deadline - 5.0:
        improved = _local_swap(best_solution, block_t_indices, n_t, n_k, t_to_blocks,
                               start_time, deadline, seed=7)
        if len(improved) < len(best_solution):
            best_solution = improved
            best_size = len(improved)

    if best_solution is None:
        return []

    return [list(k_subsets[bi]) for bi in best_solution]

# EVOLVE-BLOCK-END
