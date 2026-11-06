import itertools
import random
import time
import math
import numpy as np
import sys

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

# Precompute per-block list of covered subset indices (35 each) for fast iteration in local search
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
weights_np = np.array(weights, dtype=np.float64)
weights_uniform = np.ones(N_SUBSETS, dtype=np.float64)

# Precompute coverage matrix as numpy array for fast vectorized greedy
# coverage_matrix[i, j] = 1 if block i covers subset j
# 50388 * 3876 is ~195 million elements. As int8, that's ~195 MB. Acceptable.
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bit_indices):
    for b in bits:
        coverage_matrix[i, b] = 1

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
    deadline = start_time + 115  # Safety margin before 120s kill

    def fast_greedy(weight_vector=None):
        """Vectorized weighted greedy using numpy for speed."""
        if weight_vector is None:
            weight_vector = weights_np
        
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS
        
        while remaining > 0:
            # Compute gain for each block: sum of weights of uncovered subsets it covers
            # gains = coverage_matrix @ (weight_vector * uncovered)
            # This is a matrix-vector product, very fast in numpy
            active_weights = weight_vector * uncovered
            gains = coverage_matrix @ active_weights
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
            
            if time.time() > deadline:
                break
        
        return chosen

    # 1. Try multiple greedy initializations with different weightings
    greedy_results = []
    for w_vec in [weights_np, weights_uniform]:
        if time.time() > deadline - 10:
            break
        result = fast_greedy(w_vec)
        if result:
            greedy_results.append(result)
    
    if not greedy_results:
        # Fallback if greedy failed or timed out
        greedy_results = [fast_greedy(weights_np)]

    best_greedy = min(greedy_results, key=len)
    current_blocks = best_greedy
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

    # 2. Local Search
    no_improve_count = 0
    max_no_improve = 100
    
    # Precompute for 2-for-1 swap: for each pair of subsets, list of blocks covering both
    # This is expensive to precompute fully, so we do it on the fly or cache small parts.
    # Actually, we can just intersect the lists of blocks containing the unique subsets.
    
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        # Phase 1: Remove redundant blocks
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if is_redundant(i):
                remove_block(i)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break # Restart scan after removal
        
        if removed_any:
            no_improve_count = 0
            continue

        # Phase 2: Try 1-for-1 swap
        found_swap = False
        if len(current_blocks) > 0:
            # Sample a few blocks to try swapping
            n_sample = min(20, len(current_blocks))
            sampled = random.sample(range(len(current_blocks)), n_sample)
            
            for i in sampled:
                if time.time() > deadline:
                    break
                idx_i = current_blocks[i]
                # Find unique subsets covered by this block
                uncovered_bits = [bit for bit in block_bit_indices[idx_i] if counts[bit] == 1]
                
                if not uncovered_bits:
                    # This block is redundant (should have been caught in Phase 1, but just in case)
                    remove_block(i)
                    removed_any = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    break
                    
                # If there are few unique subsets, find a replacement
                if len(uncovered_bits) <= 5:
                    # Find blocks that cover ALL these unique subsets
                    # Start with the list of blocks containing the first subset
                    candidates = set(blocks_containing_subset[uncovered_bits[0]])
                    for b in uncovered_bits[1:]:
                        if not candidates:
                            break
                        candidates &= set(blocks_containing_subset[b])
                    
                    # Exclude the current block
                    candidates.discard(idx_i)
                    
                    if candidates:
                        # Pick a random candidate
                        b_new = random.choice(list(candidates))
                        # Check if this swap improves things (it should, as it covers the same uniques)
                        # Actually, we just swap. The number of blocks stays same.
                        # We need to ensure the new block doesn't break coverage of other subsets?
                        # No, we only remove uniques of the old block. The new block covers them.
                        # Other subsets covered by old block are still covered by others (count > 1).
                        remove_block(i)
                        add_block(b_new)
                        found_swap = True
                        break

        if found_swap:
            no_improve_count = 0
            continue

        # Phase 3: Try 2-for-1 swap
        # This is harder. We pick two blocks, find the union of their unique subsets, 
        # and see if a single block covers all of them.
        found_2for1 = False
        if len(current_blocks) >= 2:
            # Sample pairs
            n_pairs = 50
            for _ in range(n_pairs):
                if time.time() > deadline:
                    break
                i1, i2 = random.sample(range(len(current_blocks)), 2)
                idx1 = current_blocks[i1]
                idx2 = current_blocks[i2]
                
                # Find subsets uniquely covered by block 1 (count==1) or block 2 (count==1)
                # If a subset is covered by both and count==2, it becomes uncovered if we remove both.
                # So we need to cover:
                # 1. Subsets where count==1 and covered by B1
                # 2. Subsets where count==1 and covered by B2
                # 3. Subsets where count==2 and covered by BOTH B1 and B2
                
                req_bits = []
                for bit in block_bit_indices[idx1]:
                    if counts[bit] == 1:
                        req_bits.append(bit)
                    elif counts[bit] == 2 and (all_masks[idx2] >> bit) & 1:
                        req_bits.append(bit)
                
                for bit in block_bit_indices[idx2]:
                    if counts[bit] == 1:
                        # Avoid duplicates if bit was already added (e.g. if it was in req_bits from B1? 
                        # If count==1, it can't be covered by B2 if it's in B1's unique set? 
                        # If it's in B1 and count==1, it's NOT in B2. So no overlap in count==1 case.)
                        # If count==2 and covered by both, it was added in first loop.
                        # So we can just add count==1 from B2.
                        req_bits.append(bit)
                    elif counts[bit] == 2 and (all_masks[idx1] >> bit) & 1:
                        # Already added in first loop
                        pass

                if not req_bits:
                    # Both blocks are redundant?
                    remove_block(i1)
                    remove_block(i2)
                    found_2for1 = True
                    break
                    
                if len(req_bits) <= 6: # Limit to small sets to keep intersection fast
                    candidates = set(blocks_containing_subset[req_bits[0]])
                    for b in req_bits[1:]:
                        if not candidates:
                            break
                        candidates &= set(blocks_containing_subset[b])
                    
                    candidates.discard(idx1)
                    candidates.discard(idx2)
                    
                    if candidates:
                        b_new = random.choice(list(candidates))
                        # Perform swap
                        if i1 > i2:
                            remove_block(i1)
                            remove_block(i2)
                        else:
                            remove_block(i2)
                            remove_block(i1)
                        add_block(b_new)
                        found_2for1 = True
                        break

        if found_2for1:
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue

        # No improvement found
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            # Perturbation: Add a random block
            c = random.randint(0, N_BLOCKS - 1)
            if c not in current_blocks:
                add_block(c)
            no_improve_count = 0
            max_no_improve = min(500, max_no_improve * 2)

    # Final cleanup
    return [list(all_blocks[i]) for i in best_blocks]
