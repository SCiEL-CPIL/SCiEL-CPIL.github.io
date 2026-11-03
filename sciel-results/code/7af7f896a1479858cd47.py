import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    points = list(range(v))
    n_t = math.comb(v, t)

    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}

    # Precompute all possible k-blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    # Precompute t-subset indices within each block
    # For a block of size k, there are C(k, t) t-subsets
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)

    # block_t_indices[i] is a list of t-subset indices covered by block i
    block_t_indices = [None] * n_blocks
    # block_masks[i] is an integer bitmask where bit j is 1 if block i covers t_subset j
    block_masks = [0] * n_blocks

    for i, block in enumerate(all_blocks):
        indices = []
        mask = 0
        for lc in t_in_k:
            ts = tuple(block[p] for p in lc)
            idx = t_subset_index[ts]
            indices.append(idx)
            mask |= (1 << idx)
        block_t_indices[i] = indices
        block_masks[i] = mask

    # Reverse index: t_to_blocks[j] = list of block indices that cover t_subset j
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_blocks):
        for idx in block_t_indices[i]:
            t_to_blocks[idx].append(i)

    def greedy_cover(rng=None):
        """Greedy set cover with optional random tie-breaking."""
        uncovered_mask = (1 << n_t) - 1
        selected = []

        while uncovered_mask:
            if time.time() > deadline:
                break

            best_block_idx = -1
            best_score = 0
            candidates = []

            for i in range(n_blocks):
                bm = block_masks[i]
                inter = uncovered_mask & bm
                if inter == 0:
                    continue
                gain = inter.bit_count()

                if gain > best_score:
                    best_score = gain
                    best_block_idx = i
                    candidates = [i]
                elif gain == best_score and gain > 0:
                    candidates.append(i)

            if best_block_idx == -1:
                break

            if rng is not None and len(candidates) > 1:
                best_block_idx = rng.choice(candidates)

            selected.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]

        return selected

    def local_search(selected, rng=None):
        """Aggressive local search: removal + swap-based improvement + 2-for-1."""
        if not selected:
            return selected

        selected = list(selected)
        selected_set = set(selected)

        # Initialize counts
        counts = [0] * n_t
        for b_idx in selected:
            for idx in block_t_indices[b_idx]:
                counts[idx] += 1

        # Phase 1: Iterative removal of redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for b_idx in selected:
                redundant = True
                for idx in block_t_indices[b_idx]:
                    if counts[idx] <= 1:
                        redundant = False
                        break

                if redundant:
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    selected_set.discard(b_idx)
                    changed = True
                else:
                    new_selected.append(b_idx)
            selected = new_selected

        # Phase 2: Swap-based local search (1-swap) with diversity heuristic
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = list(selected)
            new_set = set(new_selected)

            # Recompute counts
            counts = [0] * n_t
            for b_idx in new_selected:
                for idx in block_t_indices[b_idx]:
                    counts[idx] += 1

            # Compute co-occurrence frequencies for diversity heuristic
            # point_freq[p] = number of selected blocks containing point p
            point_freq = [0] * v
            for b_idx in new_selected:
                for p in all_blocks[b_idx]:
                    point_freq[p] += 1

            for pos, b_idx in enumerate(list(new_selected)):
                if time.time() > deadline:
                    break

                unique_indices = [idx for idx in block_t_indices[b_idx] if counts[idx] == 1]

                if not unique_indices:
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    new_selected.remove(b_idx)
                    new_set.discard(b_idx)
                    improved = True
                    break

                # Find candidates that cover all unique_indices
                candidates = t_to_blocks[unique_indices[0]]
                for ui in unique_indices[1:]:
                    candidates = [c for c in candidates if (block_masks[c] >> ui) & 1]

                if not candidates:
                    continue

                # Score candidates by diversity: prefer blocks that cover points with low frequency
                best_swap = -1
                best_score = -1
                for cand in candidates:
                    if cand in new_set:
                        continue
                    # Diversity score: sum of inverse frequencies of points in candidate block
                    div_score = sum(1.0 / (point_freq[p] + 1) for p in all_blocks[cand])
                    if div_score > best_score:
                        best_score = div_score
                        best_swap = cand

                if best_swap != -1:
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    for idx in block_t_indices[best_swap]:
                        counts[idx] += 1
                    new_selected.remove(b_idx)
                    new_selected.append(best_swap)
                    new_set.discard(b_idx)
                    new_set.add(best_swap)
                    improved = True
                    break

            selected = new_selected

        # Phase 3: 2-for-1 replacement (replace two blocks with one if possible)
        if len(selected) >= 2 and time.time() < deadline:
            improved2 = True
            while improved2 and time.time() < deadline:
                improved2 = False
                n_sel = len(selected)
                # Try a limited number of pairs
                pairs_to_try = min(50, n_sel * (n_sel - 1) // 2)
                rng_local = random.Random()
                for _ in range(pairs_to_try):
                    if time.time() > deadline:
                        break
                    i1, i2 = rng_local.sample(range(n_sel), 2)
                    b1, b2 = selected[i1], selected[i2]

                    # Find t-subsets uniquely covered by b1 or b2
                    counts_temp = [0] * n_t
                    for b_idx in selected:
                        for idx in block_t_indices[b_idx]:
                            counts_temp[idx] += 1

                    unique_to_b1 = [idx for idx in block_t_indices[b1] if counts_temp[idx] == 1 and idx not in block_t_indices[b2]]
                    unique_to_b2 = [idx for idx in block_t_indices[b2] if counts_temp[idx] == 1 and idx not in block_t_indices[b1]]
                    all_unique = unique_to_b1 + unique_to_b2

                    if not all_unique:
                        continue

                    # Find a block that covers all unique t-subsets
                    candidates = t_to_blocks[all_unique[0]]
                    for ui in all_unique[1:]:
                        candidates = [c for c in candidates if (block_masks[c] >> ui) & 1]

                    candidates = [c for c in candidates if c not in set(selected)]

                    if candidates:
                        new_b = candidates[0]
                        # Perform the replacement
                        for idx in block_t_indices[b1]:
                            counts_temp[idx] -= 1
                        for idx in block_t_indices[b2]:
                            counts_temp[idx] -= 1
                        for idx in block_t_indices[new_b]:
                            counts_temp[idx] += 1

                        # Update selected
                        selected.remove(b1)
                        selected.remove(b2)
                        selected.append(new_b)
                        improved2 = True
                        break

        return selected

    # Multi-restart with different strategies
    best_solution = None
    best_len = float('inf')

    # Strategy 1: Simple greedy
    sol1 = greedy_cover()
    sol1 = local_search(sol1)
    if sol1:
        best_solution = sol1
        best_len = len(sol1)

    # Strategy 2: Randomized greedy with multiple trials
    for trial in range(8):
        if time.time() > deadline - 20:
            break

        rng = random.Random(trial * 137 + 42)
        sol = greedy_cover(rng)
        if not sol:
            break

        sol = local_search(sol, rng)
        if sol and len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)

    # Final: Try to improve best_solution with more aggressive local search
    if best_solution and time.time() < deadline - 5:
        best_solution = local_search(best_solution)
        if time.time() < deadline - 2:
            best_solution = local_search(best_solution)

    # Convert indices back to blocks
    if best_solution:
        return [list(all_blocks[i]) for i in best_solution]
    else:
        return []
