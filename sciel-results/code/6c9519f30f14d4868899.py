import itertools
import time
import random
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets
all_blocks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
# Incidence matrix: incidence[b, s] is 1 if subset s is in block b
# Using float32 or int8 to save memory, but float32 is good for dot products
incidence = np.zeros((1, N_SUBSETS), dtype=np.float32) # Placeholder to get size, will resize

# First pass: collect blocks and subset memberships
block_subset_indices = []
for block in itertools.combinations(range(V), K):
    subs = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        subs.append(idx)
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    block_subset_indices.append(subs)

N_BLOCKS = len(all_blocks)

# Fill incidence matrix
incidence = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.float32)
for b_idx, subs in enumerate(block_subset_indices):
    for s_idx in subs:
        incidence[b_idx, s_idx] = 1.0

# Precompute list of subset indices for each block for faster local search checks
block_subset_lists = [np.array(subs, dtype=np.int32) for subs in block_subset_indices]

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 118  # Safety margin before the 150s kill

    # 1. Greedy Initialization (vectorized)
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    uncovered = np.ones(N_SUBSETS, dtype=np.float32) # 1.0 if uncovered, 0.0 if covered
    current_blocks = []
    current_block_set = set()

    def add_block(idx):
        current_blocks.append(idx)
        current_block_set.add(idx)
        # Update counts and uncovered
        subs = block_subset_lists[idx]
        counts[subs] += 1
        # If count becomes 1, it was 0, so it's now covered
        # If count was >0, it was already covered
        # We need to update 'uncovered' where counts[subs] == 1
        # Actually, simpler: recalculate uncovered for these subsets
        # counts[subs] is updated.
        # For each sub in subs:
        #   if counts[sub] == 1: uncovered[sub] = 0.0
        #   if counts[sub] == 0: uncovered[sub] = 1.0 (should not happen on add)
        for s in subs:
            if counts[s] == 1:
                uncovered[s] = 0.0

    def remove_block(idx):
        current_block_set.remove(idx)
        pos = current_blocks.index(idx)
        current_blocks[pos] = current_blocks[-1]
        current_blocks.pop()
        subs = block_subset_lists[idx]
        counts[subs] -= 1
        for s in subs:
            if counts[s] == 0:
                uncovered[s] = 1.0

    def is_redundant(idx):
        subs = block_subset_lists[idx]
        # Redundant if all subsets in this block are covered by at least one other block
        # i.e., counts[s] >= 2 for all s in subs
        # Since we are checking if we can remove idx, we look at current counts.
        # If any count is 1, removing idx makes it 0 (uncovered).
        return np.all(counts[subs] >= 2)

    # Greedy loop using vectorized gain
    # Gain of block b is sum(incidence[b] * uncovered)
    # We can compute all gains at once: gains = incidence @ uncovered
    while np.any(uncovered > 0):
        if time.time() > deadline - 60:
            break
        gains = incidence @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        add_block(best_idx)

    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # 2. Local Search with Perturbation (ILS style)
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline - 3:
        # Phase 1: Prune redundant blocks
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)

        for i in indices:
            if time.time() > deadline - 3:
                break
            idx = current_blocks[i]
            if is_redundant(idx):
                remove_block(idx)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break

        if removed_any:
            no_improve_count = 0
            max_no_improve = max(50, max_no_improve // 2)
            continue

        # Phase 2: Try 1-for-1 swap
        if not current_blocks:
            break

        found_swap = False
        i = random.randint(0, len(current_blocks) - 1)
        idx = current_blocks[i]

        subs = block_subset_lists[idx]
        # Find unique subsets covered by idx (count == 1)
        unique_mask = (counts[subs] == 1)
        unique_bits = subs[unique_mask]

        if len(unique_bits) == 0:
            # Can just remove
            remove_block(idx)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue
        else:
            if len(unique_bits) <= 4:
                # Find candidates that cover all unique_bits
                # Sort unique_bits by scarcity
                unique_bits_list = list(unique_bits)
                unique_bits_list.sort(key=lambda b: len(blocks_containing_subset[b]))
                candidates = set(blocks_containing_subset[unique_bits_list[0]])
                for b in unique_bits_list[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])
                valid_candidates = [c for c in candidates if c not in current_block_set]
                if valid_candidates:
                    new_idx = random.choice(valid_candidates)
                    remove_block(idx)
                    add_block(new_idx)
                    found_swap = True
                    no_improve_count = 0

        if not found_swap:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: either remove random blocks + regreedy, or add a random block
                mode = random.random()
                if mode < 0.6:
                    n_remove = random.randint(3, 6)
                    to_remove = random.sample(current_blocks, min(n_remove, len(current_blocks)))
                    for idx2 in to_remove:
                        remove_block(idx2)
                    # Re-greedy repair
                    if time.time() < deadline - 8:
                        while np.any(uncovered > 0):
                            if time.time() > deadline - 3:
                                break
                            gains = incidence @ uncovered
                            bi = int(np.argmax(gains))
                            if gains[bi] == 0:
                                break
                            add_block(bi)
                else:
                    # add a random block not already present, then prune
                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in current_block_set:
                        add_block(c)
                        # immediate prune pass
                        indices2 = list(range(len(current_blocks)))
                        random.shuffle(indices2)
                        for ii in indices2:
                            if time.time() > deadline - 3:
                                break
                            if is_redundant(current_blocks[ii]):
                                remove_block(current_blocks[ii])
                                if len(current_blocks) < best_len:
                                    best_len = len(current_blocks)
                                    best_blocks = current_blocks.copy()
                                break
                no_improve_count = 0
                max_no_improve = min(200, max_no_improve * 2)

    # Final cleanup: ensure we return the best found
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            idx = current_blocks[i]
            if is_redundant(idx):
                remove_block(idx)
                changed = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()

    return [list(all_blocks[i]) for i in best_blocks]
