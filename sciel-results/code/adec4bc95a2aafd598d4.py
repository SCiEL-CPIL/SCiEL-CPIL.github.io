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

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute masks as numpy arrays for faster operations? 
# Bitmasks are better for Python integers. 
# Let's stick to Python integers for masks.

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy
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
    deadline = start_time + 115  # Safety margin

    # 1. Greedy Initialization
    # We use a weighted greedy: gain = sum(1 / (count[sub] + 1)) for sub in block \ covered
    # This tends to pick blocks that cover "rare" subsets first, which often leads to better solutions.
    
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
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

    # Greedy Loop
    uncovered_mask = (1 << N_SUBSETS) - 1
    
    # To speed up greedy, we can precompute for each block the list of subsets it covers.
    # block_subsets[i] = list of subset indices covered by block i
    block_subsets = []
    for i in range(N_BLOCKS):
        m = all_masks[i]
        subs = []
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            subs.append(bit_idx)
            mm ^= lsb
        block_subsets.append(subs)

    while uncovered_mask:
        best_idx = -1
        best_gain = 0.0
        
        # Optimization: If time is running out, pick the first available block with gain > 0
        # But let's try to be smart.
        
        for i in range(N_BLOCKS):
            m = all_masks[i]
            new_bits = m & uncovered_mask
            if not new_bits:
                continue
            
            # Calculate weighted gain
            # gain = sum(1 / (counts[sub] + 1))
            gain = 0.0
            mm = new_bits
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                gain += 1.0 / (counts[bit_idx] + 1)
                mm ^= lsb
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1:
            break
        
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
        # Time check
        if time.time() > deadline - 10:
            break

    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search
    # Strategy:
    # - Try to remove redundant blocks.
    # - If no redundant blocks, try to swap a block with another that allows removal of 2 blocks?
    #   Or simpler: Pick a block, remove it. If coverage is broken, find a block that fixes it.
    #   If the fixing block is different, we have a swap.
    #   If the fixing block is the same, it was redundant (handled above).
    #   If no fixing block exists, revert.
    
    # Enhanced Local Search:
    # 1. Prune
    # 2. If stuck, try 1-swap: remove block i, if invalid, find block j that restores validity.
    #    To make it a net improvement, we need to remove i and add j, and then prune more.
    #    Actually, if we remove i and add j, size stays same. We need to enable a prune.
    #    So: Remove i. If invalid, find j that covers the missing bits. Add j. 
    #    Then check if any block (including j or others) is now redundant.
    
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline:
        # Phase 1: Prune
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
                break
        
        if removed_any:
            no_improve_count = 0
            continue
            
        # Phase 2: Swap Search
        # Try to find a swap that leads to a reduction in size.
        # We pick a random block i to remove.
        # Find the subsets that become uncovered (unique to i).
        # If none, it was redundant (should have been caught).
        # If some, we need to add a block j that covers ALL of them.
        # After adding j, we try to prune.
        
        if len(current_blocks) == 0:
            break
            
        i = random.randint(0, len(current_blocks) - 1)
        
        # Calculate unique bits for i
        m_i = current_masks[i]
        unique_bits = []
        mm = m_i
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                unique_bits.append(bit_idx)
            mm ^= lsb
            
        if not unique_bits:
            # Redundant, but maybe is_redundant failed due to order? Unlikely.
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue
            
        # If too many unique bits, swap is unlikely to be useful or easy to find.
        if len(unique_bits) > 3:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: Add a random block that covers many unique bits of random other blocks?
                # Or just add a random block and prune.
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                    no_improve_count = 0
                    max_no_improve = min(50, max_no_improve * 2)
                continue
        else:
            # Find candidates for j
            # j must cover all unique_bits
            # Intersect blocks_containing_subset for each unique_bit
            candidates = blocks_containing_subset[unique_bits[0]]
            for b in unique_bits[1:]:
                if not candidates:
                    break
                # Optimization: Use set intersection
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
            
            if candidates:
                # Exclude i
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    # Pick one randomly
                    j = random.choice(valid_candidates)
                    
                    # Perform swap: remove i, add j
                    remove_block(i)
                    add_block(j)
                    
                    # Now try to prune
                    pruned = False
                    indices2 = list(range(len(current_blocks)))
                    random.shuffle(indices2)
                    for idx in indices2:
                        if is_redundant(idx):
                            remove_block(idx)
                            pruned = True
                            if len(current_blocks) < best_len:
                                best_len = len(current_blocks)
                                best_blocks = current_blocks.copy()
                            break
                            
                    if pruned:
                        no_improve_count = 0
                        continue
                    else:
                        # No improvement, revert? Or keep?
                        # Keeping might be neutral. Reverting is safer to explore other swaps.
                        # But if we revert, we are back to square one.
                        # Let's revert to maintain the invariant of "searching for improvement".
                        remove_block(j)
                        add_block(i)
                        no_improve_count += 1
                        if no_improve_count > max_no_improve:
                             # Perturbation
                             c = random.randint(0, N_BLOCKS - 1)
                             if c not in current_blocks:
                                 add_block(c)
                                 no_improve_count = 0
                                 max_no_improve = min(50, max_no_improve * 2)
                        continue

    # Final cleanup
    return [list(all_blocks[i]) for i in best_blocks]
