import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s

    # Precomputation
    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # 2. Enumerate all k-subsets and compute their coverage bitmask
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)

    # masks[i] is an integer bitmask where bit j is 1 if k_subset i covers t_subset j
    masks = [0] * num_k_subsets
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        masks[i] = mask

    # Helper to count bits
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')

    # Run multiple random restarts with different strategies
    num_restarts = 50000  # We will limit by time

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        # Track coverage counts for each t-subset to enable efficient local search
        coverage_count = [0] * num_t_subsets

        while uncovered:
            if time.time() > deadline:
                break

            max_gain = 0
            candidates = []

            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)

            if max_gain == 0:
                break

            # Pick a random candidate to add diversity
            chosen_idx = random.choice(candidates)
            current_blocks.append(k_subsets[chosen_idx])
            used_blocks.add(chosen_idx)
            # Update coverage counts
            mask = masks[chosen_idx]
            for j in range(num_t_subsets):
                if mask & (1 << j):
                    coverage_count[j] += 1
            uncovered &= ~mask

        # Local search: try to remove or swap blocks to reduce count
        if uncovered == 0:
            # Try to remove redundant blocks
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                changed = False
                for i in range(len(current_blocks) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    # Check if block i is redundant
                    old_idx = k_subsets.index(current_blocks[i])
                    old_mask = masks[old_idx]
                    redundant = True
                    for j in range(num_t_subsets):
                        if old_mask & (1 << j):
                            if coverage_count[j] <= 1:
                                redundant = False
                                break
                    if redundant:
                        # Remove block i
                        for j in range(num_t_subsets):
                            if old_mask & (1 << j):
                                coverage_count[j] -= 1
                        current_blocks.pop(i)
                        used_blocks.discard(old_idx)
                        changed = True
                        break

            # Try swapping: replace one block with another
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                for i in range(len(current_blocks)):
                    if time.time() > deadline:
                        break
                    old_idx = k_subsets.index(current_blocks[i])
                    old_mask = masks[old_idx]
                    # Find t-subsets only covered by this block
                    only_covered = 0
                    for j in range(num_t_subsets):
                        if old_mask & (1 << j) and coverage_count[j] == 1:
                            only_covered |= (1 << j)
                    if only_covered == 0:
                        # Block is redundant
                        for j in range(num_t_subsets):
                            if old_mask & (1 << j):
                                coverage_count[j] -= 1
                        current_blocks.pop(i)
                        used_blocks.discard(old_idx)
                        improved = True
                        break
                    # Try to find a block that covers all of only_covered
                    for j in range(num_k_subsets):
                        if j in used_blocks:
                            continue
                        if (masks[j] & only_covered) == only_covered:
                            # Swap: remove old, add new
                            for jj in range(num_t_subsets):
                                if old_mask & (1 << jj):
                                    coverage_count[jj] -= 1
                            new_mask = masks[j]
                            for jj in range(num_t_subsets):
                                if new_mask & (1 << jj):
                                    coverage_count[jj] += 1
                            current_blocks[i] = k_subsets[j]
                            used_blocks.discard(old_idx)
                            used_blocks.add(j)
                            improved = True
                            break
                    if improved:
                        break

        if uncovered == 0 and len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]

    if best_blocks is None:
        # Fallback to simple greedy if randomization failed (unlikely)
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        while uncovered:
            max_gain = 0
            best_block_idx = -1
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
            if max_gain == 0:
                break
            current_blocks.append(k_subsets[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered &= ~masks[best_block_idx]
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks
