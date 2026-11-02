import itertools
import random
import time
import numpy as np

# Precomputation
V = 19
K = 8
T = 5

# Map 5-subsets to indices
all_5_subs = list(itertools.combinations(range(V), T))
sub_index = {s: i for i, s in enumerate(all_5_subs)}
N_SUBS = len(all_5_subs) # 11628

# Precompute coverage for all 8-blocks
all_8_blocks = list(itertools.combinations(range(V), K))
block_coverage = []
block_bits = [] # Bitset representation for fast operations
for b in all_8_blocks:
    cov = [sub_index[s] for s in itertools.combinations(b, T)]
    block_coverage.append(cov)
    bits = 0
    for s_idx in cov:
        bits |= (1 << s_idx)
    block_bits.append(bits)

# Map from 5-subset to list of block indices containing it
sub_to_blocks = [[] for _ in range(N_SUBS)]
for b_idx, b in enumerate(all_8_blocks):
    for s_idx in block_coverage[b_idx]:
        sub_to_blocks[s_idx].append(b_idx)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Greedy Construction
    def greedy_construct():
        uncovered = [True] * N_SUBS
        blocks = []
        while True:
            # Find an uncovered subset
            target_idx = -1
            for i in range(N_SUBS):
                if uncovered[i]:
                    target_idx = i
                    break
            if target_idx == -1:
                break
            
            # Get all blocks covering this subset
            candidates = sub_to_blocks[target_idx]
            
            # Pick the block that covers the most uncovered subsets
            best_block = -1
            max_gain = -1
            for b_idx in candidates:
                gain = 0
                for s_idx in block_coverage[b_idx]:
                    if uncovered[s_idx]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
            
            # Add the block
            blocks.append(best_block)
            for s_idx in block_coverage[best_block]:
                uncovered[s_idx] = False
        
        return blocks

    # Initial Solution
    current_blocks = greedy_construct()
    
    # Local Search with Bitsets
    def local_search(blocks):
        counts = [0] * N_SUBS
        for b_idx in blocks:
            for s_idx in block_coverage[b_idx]:
                counts[s_idx] += 1
        
        # unique_mask: bit i is 1 if counts[i] == 1
        unique_mask = 0
        for i in range(N_SUBS):
            if counts[i] == 1:
                unique_mask |= (1 << i)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Pass 1: Remove redundant blocks
            random.shuffle(blocks)
            for b_idx in blocks[:]:
                # Check redundancy using bitset
                if (block_bits[b_idx] & unique_mask) == 0:
                    # Remove
                    for s_idx in block_coverage[b_idx]:
                        old_count = counts[s_idx]
                        counts[s_idx] = old_count - 1
                        # Update unique_mask
                        if old_count == 1:
                            # Becomes 0, clear bit
                            unique_mask &= ~(1 << s_idx)
                        elif old_count == 2:
                            # Becomes 1, set bit
                            unique_mask |= (1 << s_idx)
                    blocks.remove(b_idx)
                    improved = True
            
            # Pass 2: 1-for-1 Swaps
            if time.time() > deadline - 10:
                break

            random.shuffle(blocks)
            for b_idx in blocks[:]:
                if time.time() > deadline - 5.0:
                    break
                
                # Find unique subsets of b_idx
                U_b = block_bits[b_idx] & unique_mask
                if U_b == 0:
                    continue # Redundant, should have been removed
                
                # Pick a representative subset from U_b to get candidates
                # Find lowest set bit
                # s_0 = (U_b & -U_b).bit_length() - 1
                # But we need the index.
                # Let's just iterate through block_coverage[b_idx] and find one with count 1
                rep_s = -1
                for s_idx in block_coverage[b_idx]:
                    if counts[s_idx] == 1:
                        rep_s = s_idx
                        break
                
                if rep_s == -1:
                    continue
                
                candidates = sub_to_blocks[rep_s]
                
                # Find a candidate c != b_idx that covers all of U_b
                best_c = -1
                # We want a candidate that covers U_b.
                # We can also prefer one that minimizes variance or something, 
                # but for now, any valid swap that allows further reduction is good.
                # To make it deterministic-ish or random, we can shuffle candidates?
                # Or just pick the first one.
                # Let's try to pick one that covers the most *other* unique subsets? 
                # No, just valid swap.
                
                for c_idx in candidates:
                    if c_idx == b_idx:
                        continue
                    # Check if c covers U_b
                    if (block_bits[c_idx] & U_b) == U_b:
                        best_c = c_idx
                        break
                
                if best_c != -1:
                    # Perform swap
                    # Remove b_idx
                    for s_idx in block_coverage[b_idx]:
                        old_count = counts[s_idx]
                        counts[s_idx] = old_count - 1
                        if old_count == 1:
                            unique_mask &= ~(1 << s_idx)
                        elif old_count == 2:
                            unique_mask |= (1 << s_idx)
                    
                    # Add best_c
                    for s_idx in block_coverage[best_c]:
                        old_count = counts[s_idx]
                        counts[s_idx] = old_count + 1
                        if old_count == 0:
                            # Becomes 1, set bit
                            unique_mask |= (1 << s_idx)
                        elif old_count == 1:
                            # Becomes 2, clear bit
                            unique_mask &= ~(1 << s_idx)
                    
                    blocks.remove(b_idx)
                    blocks.append(best_c)
                    improved = True
                    break # Restart loop after swap
            
        return blocks

    current_blocks = local_search(current_blocks)
    
    # Final Prune
    def final_prune(blocks):
        if not blocks:
            return blocks
        counts = [0] * N_SUBS
        for b_idx in blocks:
            for s_idx in block_coverage[b_idx]:
                counts[s_idx] += 1
        
        new_blocks = []
        for b_idx in blocks:
            is_redundant = True
            for s_idx in block_coverage[b_idx]:
                if counts[s_idx] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                for s_idx in block_coverage[b_idx]:
                    counts[s_idx] -= 1
            else:
                new_blocks.append(b_idx)
                
        return new_blocks

    current_blocks = final_prune(current_blocks)
    
    result = [list(all_8_blocks[b_idx]) for b_idx in current_blocks]
    
    return result
