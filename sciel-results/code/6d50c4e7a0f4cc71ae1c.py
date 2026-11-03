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

    def dual_guided_cover(uncovered, selected_blocks, max_time_margin=5.0):
        """Primal-dual heuristic: iteratively raise duals for uncovered sets,
        select blocks when their dual sum exceeds a threshold."""
        dual = np.zeros(n_t, dtype=np.float64)
        block_dual_sum = np.zeros(n_k, dtype=np.float64)
        selected = set()
        
        # Initial dual values based on rarity
        t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
        t_counts[t_counts == 0] = 1
        dual = 1.0 / t_counts.astype(np.float64)
        block_dual_sum = (dual[block_t_indices]).sum(axis=1)
        
        while True:
            if time.time() > deadline - max_time_margin:
                break
            if not uncovered.any():
                break
            
            # Find uncovered t-subsets
            uncov_idx = np.where(uncovered)[0]
            if len(uncov_idx) == 0:
                break
            
            # Raise duals for uncovered sets
            # Use adaptive step: larger for very uncovered, smaller for recently covered
            step = 0.1 + 0.5 * (1.0 / (1.0 + np.log1p(len(selected) / 10.0)))
            dual[uncov_idx] += step
            
            # Update block dual sums for blocks covering newly raised duals
            # For efficiency, only update blocks that cover any uncovered set
            # Find blocks that cover at least one uncovered set
            uncov_mask = uncovered[block_t_indices]
            # For each block, count how many of its t-subsets are uncovered
            new_block_dual_sum = (dual[block_t_indices] * uncov_mask).sum(axis=1)
            
            # Select blocks where dual sum is high relative to cost (1 per block)
            # A block is "tight" if its dual sum >= 1
            tight_blocks = np.where(new_block_dual_sum >= 1.0)[0]
            
            if len(tight_blocks) == 0:
                # No tight blocks, pick the one with highest dual sum
                best_block = int(np.argmax(new_block_dual_sum))
                selected.add(best_block)
                selected_blocks.append(best_block)
                uncovered[block_t_indices[best_block]] = False
            else:
                # Among tight blocks, pick the one covering most uncovered sets
                cov_counts = uncov_mask[tight_blocks].sum(axis=1)
                best_in_tight = tight_blocks[np.argmax(cov_counts)]
                selected.add(best_in_tight)
                selected_blocks.append(best_in_tight)
                uncovered[block_t_indices[best_in_tight]] = False
        
        return uncovered

    def greedy_fill(uncovered, selected_blocks, max_time_margin=5.0):
        """Standard weighted greedy as fallback."""
        t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
        t_counts[t_counts == 0] = 1
        t_weights = 1.0 / t_counts.astype(np.float64)
        
        while True:
            if time.time() > deadline - max_time_margin:
                break
            if not uncovered.any():
                break

            gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            best_idx = rng.choice(candidates)
            selected_blocks.append(best_idx)
            uncovered[block_t_indices[best_idx]] = False

        return uncovered

    restart_count = 0
    while time.time() < deadline:
        restart_count += 1

        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []

        # Choose method: dual-guided or standard greedy
        use_dual = (restart_count % 2 == 0)
        if use_dual:
            uncovered = dual_guided_cover(uncovered, selected_blocks)
        else:
            uncovered = greedy_fill(uncovered, selected_blocks)

        # Check if we finished within time limit
        if time.time() > deadline - 5.0:
            break

        # Post-optimization: Remove redundant blocks
        if selected_blocks:
            remaining = remove_redundant(list(selected_blocks), block_t_indices, n_t)

            if len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]

            # Local Search: Shake and Repair
            if time.time() < deadline - 10.0 and len(remaining) > 0:
                shake_size = max(1, len(remaining) // 10)
                shake_indices = rng.sample(range(len(remaining)), min(shake_size, len(remaining)))
                to_remove = set(remaining[i] for i in shake_indices)
                new_remaining = [b for b in remaining if b not in to_remove]

                # Re-calculate uncovered based on new_remaining
                new_uncovered = np.ones(n_t, dtype=bool)
                for bi in new_remaining:
                    new_uncovered[block_t_indices[bi]] = False

                # Re-fill using dual-guided
                new_uncovered = dual_guided_cover(new_uncovered, new_remaining)

                # Re-remove redundants
                new_remaining = remove_redundant(new_remaining, block_t_indices, n_t)

                if len(new_remaining) < best_size:
                    best_size = len(new_remaining)
                    best_solution = [k_subsets[i] for i in new_remaining]

    if best_solution is None:
        return []

    return [list(b) for b in best_solution]
