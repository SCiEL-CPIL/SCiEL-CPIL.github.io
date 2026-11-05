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

# Precompute weights for weighted greedy
# Weight is inverse of frequency, normalized
max_freq = max(subset_frequencies)
weights = [1.0 / f for f in subset_frequencies]

# Precompute subset lists for each block for faster iteration if needed
# But bit masks are faster for intersection.

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
    deadline = start_time + 118  # Safety margin
    
    best_len = float('inf')
    best_blocks = []
    
    # Run multiple iterations with different random seeds or perturbations
    # We will run a main loop that tries to improve the best found solution
    
    for restart in range(10):  # Up to 10 restarts if time permits
        if time.time() > deadline:
            break
            
        current_blocks = []
        current_masks = []
        counts = [0] * N_SUBSETS
        
        def add_block(idx):
            current_blocks.append(idx)
            current_masks.append(all_masks[idx])
            m = all_masks[idx]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                mm ^= lsb

        def remove_block(i):
            m = current_masks[i]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] -= 1
                mm ^= lsb
            # Remove from lists
            current_blocks.pop(i)
            current_masks.pop(i)

        def is_redundant(i):
            m = current_masks[i]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    return False
                mm ^= lsb
            return True

        # 1. Weighted Greedy Initialization
        uncovered_mask = (1 << N_SUBSETS) - 1
        while uncovered_mask:
            best_idx = -1
            best_gain = 0.0
            # Optimization: Only check blocks that cover at least one uncovered subset
            # But iterating all is O(N_BLOCKS). N_BLOCKS is ~50k.
            # We can precompute for each subset the list of blocks.
            # But finding the max gain requires summing weights of uncovered subsets in the block.
            # This is O(K*T) per block if we iterate subsets.
            # With bit masks, we can do:
            # gain = sum(weights[bit] for bit in (mask & uncovered_mask))
            # This is still O(N_SUBSETS) in worst case if we iterate bits.
            # However, N_SUBSETS is 3876. 50k * 3876 is too slow.
            # We need a faster gain calculation.
            
            # Alternative: Precompute for each block the list of subset indices.
            # Then gain = sum(weights[s] for s in block_subsets if uncovered[s])
            # This is O(35) per block (35 subsets per 7-block).
            # 50k * 35 = 1.75M operations per greedy step.
            # Greedy steps ~ 150. Total ~ 260M operations. Might be slow in Python.
            
            # Let's stick to the bit mask approach but optimize the inner loop.
            # Python's int.bit_count() is fast, but we need weighted sum.
            # We can precompute the weight sum for each block? No, weights are static, but uncovered_mask changes.
            
            # Let's use the list of subsets for each block.
            # Precompute block_subsets_list
            # We didn't do that in the global setup. Let's do it now if needed, or just use the mask.
            # Actually, the previous program used masks and iterated bits.
            # Let's try to speed up the greedy by only considering blocks that intersect the uncovered set.
            # But we don't have a fast way to get those blocks without scanning all.
            
            # Let's use the standard greedy with mask iteration, but break early if possible?
            # No, we need the max.
            
            # Let's assume the previous greedy was fast enough.
            # To speed up, we can use NumPy for the weight sum?
            # We can represent weights as a NumPy array.
            # But we need to sum weights of bits set in (mask & uncovered_mask).
            # This is a sparse dot product.
            
            # For now, let's use the loop, but optimize it.
            for i in range(N_BLOCKS):
                m = all_masks[i]
                mm = m & uncovered_mask
                if mm == 0:
                    continue
                gain = 0.0
                # Iterate bits
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    gain += weights[bit_idx]
                    mm ^= lsb
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_gain == 0:
                break
            add_block(best_idx)
            uncovered_mask &= ~all_masks[best_idx]
            
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks.copy()
            
        # 2. Local Search
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < deadline:
            if time.time() > deadline: break
            
            # Try to remove redundant blocks
            removed_any = False
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline: break
                if is_redundant(i):
                    remove_block(i)
                    removed_any = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    break
            
            if removed_any:
                no_improve_count = 0
                continue
                
            # If no redundant block found, try a 2-for-1 swap
            found_swap = False
            if len(current_blocks) >= 2:
                # Systematic search for 2-for-1
                # Pick a block B1
                # To make it faster, we can limit the number of attempts or use a heuristic
                # Let's try random pairs first, as systematic might be too slow
                attempts = 100
                for _ in range(attempts):
                    if time.time() > deadline: break
                    i1 = random.randint(0, len(current_blocks) - 1)
                    i2 = random.randint(0, len(current_blocks) - 1)
                    if i1 == i2: continue
                    
                    b1_idx = current_blocks[i1]
                    b2_idx = current_blocks[i2]
                    
                    # Calculate uncovered bits if we remove both
                    uncovered_bits = set()
                    
                    m1 = all_masks[b1_idx]
                    mm = m1
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        if counts[bit_idx] == 1:
                            uncovered_bits.add(bit_idx)
                        elif counts[bit_idx] == 2:
                            if all_masks[b2_idx] & (1 << bit_idx):
                                uncovered_bits.add(bit_idx)
                        mm ^= lsb
                    
                    m2 = all_masks[b2_idx]
                    mm = m2
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        if counts[bit_idx] == 1:
                            uncovered_bits.add(bit_idx)
                        mm ^= lsb

                    if not uncovered_bits:
                        continue

                    # Find a block that covers all uncovered_bits
                    bits_list = list(uncovered_bits)
                    bits_list.sort(key=lambda b: len(blocks_containing_subset[b]))
                    
                    candidates = set(blocks_containing_subset[bits_list[0]])
                    for b in bits_list[1:]:
                        if not candidates: break
                        candidates &= set(blocks_containing_subset[b])
                    
                    if candidates:
                        candidates.discard(b1_idx)
                        candidates.discard(b2_idx)
                        if candidates:
                            b3_idx = random.choice(list(candidates))
                            # Perform swap
                            if i1 > i2:
                                remove_block(i1)
                                remove_block(i2)
                            else:
                                remove_block(i2)
                                remove_block(i1)
                            add_block(b3_idx)
                            
                            if len(current_blocks) < best_len:
                                best_len = len(current_blocks)
                                best_blocks = current_blocks.copy()
                            
                            found_swap = True
                            no_improve_count = 0
                            break
                            
            if found_swap:
                continue

            # Try 1-for-1 Swap
            i = random.randint(0, len(current_blocks) - 1)
            m_i = current_masks[i]
            uncovered_bits = []
            mm = m_i
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    uncovered_bits.append(bit_idx)
                mm ^= lsb
                
            if not uncovered_bits:
                remove_block(i)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                continue
                
            if len(uncovered_bits) <= 4:
                uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                candidates = set(blocks_containing_subset[uncovered_bits[0]])
                for b in uncovered_bits[1:]:
                    if not candidates: break
                    candidates &= set(blocks_containing_subset[b])
                
                if candidates:
                    candidates.discard(current_blocks[i])
                    if candidates:
                        j = random.choice(list(candidates))
                        remove_block(i)
                        add_block(j)
                        no_improve_count = 0
                        continue
                        
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                no_improve_count = 0
                max_no_improve = min(500, max_no_improve * 2)

    # Final cleanup
    return [list(all_blocks[i]) for i in best_blocks]
