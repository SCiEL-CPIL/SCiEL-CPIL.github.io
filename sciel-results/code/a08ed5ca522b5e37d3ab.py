import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute per-block list of covered subset indices (35 each) for fast iteration
block_bit_indices = []
for mask in all_masks:
    bits = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bit_indices.append(bits)

# Weighted greedy weights: rare subsets get higher priority
weights = [1.0 / f for f in subset_frequencies]

# Precompute coverage matrix as numpy array for fast vectorized greedy
# coverage_matrix[i, j] = 1 if block i covers subset j
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bit_indices):
    for b in bits:
        coverage_matrix[i, b] = 1

# weights as numpy array
weights_np = np.array(weights, dtype=np.float64)


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
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
    deadline = start_time + 112  # Safety margin below 120s kill

    def fast_greedy():
        """Vectorized weighted greedy using numpy for speed."""
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS
        
        while remaining > 0:
            # Compute gain for each block: sum of weights of uncovered subsets it covers
            gains = coverage_matrix @ (weights_np * uncovered)
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            
            if best_gain <= 0:
                break
            
            chosen.append(best_idx)
            
            # Update uncovered
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered
        
        return chosen

    # 1. Fast greedy initialization
    current_blocks = fast_greedy()
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # Build counts (list of ints, fast)
    counts = [0] * N_SUBSETS
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def add_block(idx):
        current_blocks.append(idx)
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def remove_block(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            counts[bit] -= 1
        current_blocks.pop(pos)

    def is_redundant(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            if counts[bit] == 1:
                return False
        return True

    def find_redundant():
        for pos in range(len(current_blocks)):
            if is_redundant(pos):
                return pos
        return -1

    # 2. Local search: repeatedly remove redundant blocks, then do swaps
    no_improve = 0
    max_no_improve = 40
    current_set = set(current_blocks)

    def rebuild_set():
        nonlocal current_set
        current_set = set(current_blocks)

    while time.time() < deadline:
        # Phase A: prune all redundant blocks
        progress = True
        while progress:
            progress = False
            pos = find_redundant()
            if pos >= 0:
                remove_block(pos)
                current_set.discard(current_blocks[pos] if pos < len(current_blocks) else -1)
                # After pop, need to rebuild set
                current_set = set(current_blocks)
                progress = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve = 0
                continue

        if len(current_blocks) == 0:
            break

        # Phase B: try a 1-for-1 swap (remove one, add one that recovers uncovered)
        # Pick the block with the fewest unique covers (most "replaceable")
        best_swap_candidate = -1
        min_unique = float('inf')
        
        # Sample a subset of blocks to find good swap candidates
        n_sample = min(len(current_blocks), 30)
        sampled_indices = random.sample(range(len(current_blocks)), n_sample)
        
        for i in sampled_indices:
            idx_i = current_blocks[i]
            n_unique = 0
            for bit in block_bit_indices[idx_i]:
                if counts[bit] == 1:
                    n_unique += 1
            if 0 < n_unique < min_unique:
                min_unique = n_unique
                best_swap_candidate = i
        
        if best_swap_candidate < 0:
            # Try random
            i = random.randint(0, len(current_blocks) - 1)
            idx_i = current_blocks[i]
            uncovered_bits = [bit for bit in block_bit_indices[idx_i] if counts[bit] == 1]
            if not uncovered_bits:
                remove_block(i)
                current_set = set(current_blocks)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve = 0
                continue
            best_swap_candidate = i
            idx_i = current_blocks[i]
            uncovered_bits = [bit for bit in block_bit_indices[idx_i] if counts[bit] == 1]
        else:
            i = best_swap_candidate
            idx_i = current_blocks[i]
            uncovered_bits = [bit for bit in block_bit_indices[idx_i] if counts[bit] == 1]

        if not uncovered_bits:
            remove_block(i)
            current_set = set(current_blocks)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve = 0
            continue

        # Need a replacement block covering all uncovered_bits
        if len(uncovered_bits) > 5:
            no_improve += 1
            if no_improve > max_no_improve:
                # Perturbation: add a block that covers many currently under-covered subsets
                # Pick a random subset with count 1, add a block covering it
                under_covered = [b for b in range(N_SUBSETS) if counts[b] == 1]
                if under_covered:
                    target = random.choice(under_covered)
                    cands_for_target = blocks_containing_subset[target]
                    c = random.choice(cands_for_target)
                    if c not in current_set:
                        add_block(c)
                        current_set.add(c)
                no_improve = 0
            continue

        # Find candidate blocks covering all uncovered bits
        # Use the rarest uncovered bit to start with smallest candidate list
        uncovered_bits_sorted = sorted(uncovered_bits, key=lambda b: len(blocks_containing_subset[b]))
        cands = set(blocks_containing_subset[uncovered_bits_sorted[0]])
        for b in uncovered_bits_sorted[1:]:
            cands.intersection_update(blocks_containing_subset[b])
            if not cands:
                break

        if cands:
            cands.discard(idx_i)
            if cands:
                # Choose the candidate that maximizes additional coverage
                cands_list = list(cands)
                if len(cands_list) <= 10:
                    best_j = cands_list[0]
                    best_extra = -1
                    for j in cands_list:
                        extra = 0
                        for bit in block_bit_indices[j]:
                            if counts[bit] == 0:
                                extra += 1
                        if extra > best_extra:
                            best_extra = extra
                            best_j = j
                    j = best_j
                else:
                    j = random.choice(cands_list)
                
                remove_block(i)
                add_block(j)
                current_set = set(current_blocks)
                no_improve = 0
                continue

        no_improve += 1
        if no_improve > max_no_improve:
            # Perturbation: add 2-3 blocks covering under-covered subsets
            under_covered = [b for b in range(N_SUBSETS) if counts[b] == 1]
            if under_covered:
                for _ in range(random.randint(2, 3)):
                    target = random.choice(under_covered)
                    cands_for_target = blocks_containing_subset[target]
                    c = random.choice(cands_for_target)
                    if c not in current_set:
                        add_block(c)
                        current_set.add(c)
            else:
                for _ in range(random.randint(1, 2)):
                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in current_set:
                        add_block(c)
                        current_set.add(c)
            no_improve = 0

    # 3. If time remains, do a fresh fast-greedy restart and keep the better solution
    if time.time() < deadline - 10:
        alt = fast_greedy()
        if len(alt) < best_len:
            best_len = len(alt)
            best_blocks = alt.copy()
            # Quick prune pass
            counts2 = [0] * N_SUBSETS
            for idx in best_blocks:
                for bit in block_bit_indices[idx]:
                    counts2[bit] += 1
            changed = True
            while changed and time.time() < deadline - 3:
                changed = False
                for pos in range(len(best_blocks)):
                    idx = best_blocks[pos]
                    redundant = True
                    for bit in block_bit_indices[idx]:
                        if counts2[bit] == 1:
                            redundant = False
                            break
                    if redundant:
                        for bit in block_bit_indices[idx]:
                            counts2[bit] -= 1
                        best_blocks.pop(pos)
                        changed = True
                        if len(best_blocks) < best_len:
                            best_len = len(best_blocks)
                        break

    # Final validation: ensure all blocks are distinct and valid
    seen = set()
    final_blocks = []
    for idx in best_blocks:
        if idx not in seen:
            seen.add(idx)
            final_blocks.append(list(all_blocks[idx]))
    
    return final_blocks
