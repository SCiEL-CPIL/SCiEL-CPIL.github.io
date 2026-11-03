import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)

    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)

    # Filling the array
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]

    # 4. Build sparse coverage matrix for efficient dual updates
    # For each block, store which t-subsets it covers
    block_t_sets = [set(block_t_indices[i].tolist()) for i in range(n_k)]

    # 5. Primal-dual heuristic
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)

    def remove_redundant(remaining, block_t_indices, n_t):
        if not remaining:
            return remaining
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in remaining:
            coverage_count[block_t_indices[bi]] += 1

        changed = True
        rng.shuffle(remaining)
        while changed:
            if time.time() > deadline - 2.0:
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

    # Precompute static coverage-frequency weight (rarity) once.
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    t_weights = 1.0 / t_counts.astype(np.float64)

    def grasp_construct(uncovered, selected_blocks, alpha, max_time_margin=5.0):
        """GRASP construction: build RCL from a smooth (log-sum-exp) score
        balancing immediate coverage gain and long-term potential, then pick
        randomly from the RCL. alpha in [0,1] trades off greediness vs diversity."""
        # Smooth "potential": how many other blocks share each block's t-sets.
        # Higher = selecting this block makes more others redundant = good.
        # Approximate via squared co-occurrence of t-subsets (block_t_indices is
        # a 0/1 indicator per (block, t-in-k) slot).
        cooc = np.zeros(n_t, dtype=np.int64)
        for ts in block_t_indices.ravel():
            cooc[ts] += 1
        # potential[b] = sum over its t-sets of (cooc[t] - 1)  (other blocks covering same t)
        potential = (cooc[block_t_indices] - 1).sum(axis=1).astype(np.float64)
        pot_max = potential.max() if potential.size else 1.0
        potential_norm = potential / (pot_max if pot_max > 0 else 1.0)

        while True:
            if time.time() > deadline - max_time_margin:
                break
            if not uncovered.any():
                break

            uncov_mask = uncovered[block_t_indices]
            gain = (uncov_mask * t_weights[block_t_indices]).sum(axis=1)
            gain_max = gain.max()
            if gain_max == 0:
                break
            gain_norm = gain / gain_max

            # Smooth combined score; log-sum-exp is used only to keep a smooth
            # surrogate of the max (soft-max) for thresholding, not the objective.
            score = alpha * gain_norm + (1.0 - alpha) * potential_norm
            # Soft-max of score (temperature 0.5) as smooth upper surrogate.
            temp = 0.5
            shifted = score / temp
            shifted -= shifted.max()
            softmax = np.exp(shifted)
            logsumexp = np.log(softmax.sum())
            # RCL: blocks whose score is within a soft threshold of the soft-max.
            thr = logsumexp - 1.0  # blocks contributing most of the soft-max mass
            rcl = np.where(score >= np.exp(thr - shifted.max()))[0]
            if len(rcl) == 0:
                rcl = np.where(score == score.max())[0]
            best_idx = int(rng.choice(rcl))
            selected_blocks.append(best_idx)
            uncovered[block_t_indices[best_idx]] = False

        return uncovered

    def local_search_swap(remaining, block_t_indices, n_t, max_time_margin=8.0):
        """Swap-based local search: try replacing one block with another that
        restores any t-sets the removed block uniquely covered."""
        if not remaining:
            return remaining
        remaining = list(remaining)
        # Precompute, for each t-set, which remaining blocks cover it.
        # Use a dict of sets for speed on moderate sizes.
        cover = {}
        for bi in remaining:
            for ts in block_t_indices[bi]:
                cover.setdefault(int(ts), set()).add(bi)

        improved = True
        while improved and time.time() < deadline - max_time_margin:
            improved = False
            rng.shuffle(remaining)
            for i in range(len(remaining)):
                if time.time() > deadline - max_time_margin:
                    break
                bi = remaining[i]
                # t-sets uniquely covered by bi
                unique_ts = [int(ts) for ts in block_t_indices[bi]
                             if len(cover.get(int(ts), ())) == 1]
                if not unique_ts:
                    # bi is redundant -> remove
                    for ts in block_t_indices[bi]:
                        cover[int(ts)].discard(bi)
                    remaining.pop(i)
                    improved = True
                    break
                # Find a replacement block (not in remaining) covering all unique_ts
                cand = None
                for j in range(n_k):
                    if j == bi or j in remaining_set(remaining):
                        continue
                    ok = True
                    for ts in unique_ts:
                        if ts not in block_t_indices[j]:
                            ok = False
                            break
                    if ok:
                        cand = j
                        break
                if cand is not None:
                    # swap
                    for ts in block_t_indices[bi]:
                        cover[int(ts)].discard(bi)
                    for ts in block_t_indices[cand]:
                        cover.setdefault(int(ts), set()).add(cand)
                    remaining[i] = cand
                    improved = True
                    break
        return remaining

    def remaining_set(lst):
        return set(lst)

    restart_count = 0
    while time.time() < deadline:
        restart_count += 1

        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []

        # GRASP: vary alpha across restarts to explore the gain/potential tradeoff.
        alpha = 0.5 + 0.5 * rng.random()
        uncovered = grasp_construct(uncovered, selected_blocks, alpha)

        # Check if we finished within time limit
        if time.time() > deadline - 5.0:
            break

        # Post-optimization: Remove redundant blocks
        if selected_blocks:
            remaining = remove_redundant(list(selected_blocks), block_t_indices, n_t)

            if len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]

            # Swap-based local search
            if time.time() < deadline - 10.0 and len(remaining) > 0:
                remaining = local_search_swap(remaining, block_t_indices, n_t)
                if len(remaining) < best_size:
                    best_size = len(remaining)
                    best_solution = [k_subsets[i] for i in remaining]

    if best_solution is None:
        return []

    return [list(b) for b in best_solution]
