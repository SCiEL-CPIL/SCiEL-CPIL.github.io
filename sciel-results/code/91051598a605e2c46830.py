import itertools
import random
import time
import math
import numpy as np
from collections import defaultdict

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
weights = [1.0 / f for f in subset_frequencies]

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
    deadline = start_time + 115  # Safety margin
    
    # 1. Weighted Greedy Initialization
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

    # Weighted Greedy
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0.0
        for i in range(N_BLOCKS):
            m = all_masks[i]
            gain = 0.0
            mm = m & uncovered_mask
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
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search with Perturbation
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
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
            
        # If no redundant block found, try a swap
        if len(current_blocks) == 0: break
        
        # --- 2-for-1 Exchange ---
        found_2_1 = False
        if len(current_blocks) >= 2:
            attempts = 80
            for _ in range(attempts):
                if time.time() > deadline: break
                i1 = random.randint(0, len(current_blocks) - 1)
                i2 = random.randint(0, len(current_blocks) - 1)
                if i1 == i2: continue
                
                b1_idx = current_blocks[i1]
                b2_idx = current_blocks[i2]
                
                uncovered_bits = set()
                
                # Check subsets in b1
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
                
                # Check subsets in b2
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
                        
                        found_2_1 = True
                        no_improve_count = 0
                        break
        
        if found_2_1:
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
            c = random.randint(0, N_BLOCKS - 1)
            if c not in current_blocks:
                add_block(c)
            no_improve_count = 0
            max_no_improve = min(1000, max_no_improve * 2)

    # Final cleanup: prune redundancies from best_blocks
    # We need to reconstruct counts for best_blocks to prune safely?
    # Actually, best_blocks is a snapshot. We should prune it independently.
    # Let's create a helper to prune a list of block indices.
    
    def prune_blocks(block_indices):
        # Count coverage
        cnt = [0] * N_SUBSETS
        for idx in block_indices:
            m = all_masks[idx]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                cnt[bit_idx] += 1
                mm ^= lsb
        
        # Remove redundant
        pruned = []
        for idx in block_indices:
            m = all_masks[idx]
            redundant = True
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if cnt[bit_idx] == 1:
                    redundant = False
                    break
                mm ^= lsb
            
            if not redundant:
                pruned.append(idx)
            else:
                # Decrement counts
                mm = m
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    cnt[bit_idx] -= 1
                    mm ^= lsb
        
        return pruned

    best_blocks = prune_blocks(best_blocks)
    
    return [list(all_blocks[i]) for i in best_blocks]
