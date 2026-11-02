import itertools
import random
import numpy as np
from collections import defaultdict
import time

def construct(v, k, t):
    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map t-subset to index
    tsub_to_idx = {tuple(cs): i for i, cs in enumerate(all_tsubs)}
    
    # Precompute point masks for t-subsets
    # tsub_point_mask[j] is an int with bits set for points in all_tsubs[j]
    tsub_point_mask = [0] * num_tsubs
    for idx, cs in enumerate(all_tsubs):
        mask = 0
        for p in cs:
            mask |= (1 << p)
        tsub_point_mask[idx] = mask
            
    # Greedy Construction
    # uncovered is a list of indices of uncovered t-subsets
    # Using a set for O(1) removal
    uncovered_set = set(range(num_tsubs))
    blocks = []
    
    while uncovered_set:
        # Pick a random uncovered t-subset to start the block
        # Or the first one. Random might help diversity? 
        # Let's pick the one with smallest "degree" or just first.
        # For speed, just pick first.
        start_idx = next(iter(uncovered_set))
        start_tsub = all_tsubs[start_idx]
        
        block = list(start_tsub)
        block_set = set(block)
        block_mask = 0
        for p in block:
            block_mask |= (1 << p)
            
        # Greedily add points
        while len(block) < k:
            best_p = -1
            best_gain = -1
            
            # Calculate gain for each candidate point
            # Gain = number of uncovered t-subsets that would be covered by adding p
            # Condition: tsub_point_mask[j] & ~block_mask == (1 << p)
            
            # We can iterate through uncovered_set
            # To optimize, we can pre-compute for current block_mask, 
            # which points have high potential.
            
            # Simple approach: iterate all p
            for p in range(v):
                if p in block_set:
                    continue
                gain = 0
                # Iterate through uncovered t-subsets
                # Optimization: Only check t-subsets that contain p?
                # We don't have that index easily.
                # But we can check: if (tsub_point_mask[j] & (1<<p)) == 0, skip.
                # And if (tsub_point_mask[j] & ~block_mask) != (1<<p), skip.
                
                # This inner loop is expensive.
                # Let's try to optimize by precomputing "missing masks" for uncovered set?
                # No, block_mask changes.
                
                # Alternative: 
                # For each uncovered j, miss = tsub_point_mask[j] & ~block_mask.
                # If miss is power of 2, it contributes to gain[p].
                
                # Let's restructure:
                # For current block_mask, compute gain for all p in one pass over uncovered_set.
                # This is better.
                
                pass # Placeholder for restructure
                
            # Restructured Gain Calculation
            gains = [0] * v
            for j in uncovered_set:
                miss = tsub_point_mask[j] & ~block_mask
                if miss == 0:
                    # Should not happen if uncovered_set is correct
                    continue
                # Check if miss is a power of 2
                if miss & (miss - 1) == 0:
                    # It is a power of 2
                    p = miss.bit_length() - 1
                    if p not in block_set:
                        gains[p] += 1
                # If miss has >1 bits, it doesn't contribute to single-point gain
                # If miss has 0 bits, it's covered (should be removed)
            
            best_gain = max(gains)
            if best_gain == 0:
                # No single point adds coverage? 
                # This shouldn't happen if uncovered_set is not empty and we can add points.
                # Unless k is reached?
                # If best_gain is 0, pick any point not in block.
                for p in range(v):
                    if p not in block_set:
                        best_p = p
                        break
            else:
                best_p = gains.index(best_gain)
                
            if best_p == -1:
                break
                
            block.append(best_p)
            block_set.add(best_p)
            block_mask |= (1 << best_p)
            
            # Update uncovered_set
            # Remove all j such that tsub_point_mask[j] & ~block_mask == 0
            # Iterate through a copy? Or filter.
            # Since we are adding points, we only remove.
            # We can filter the set.
            new_uncovered = set()
            for j in uncovered_set:
                if tsub_point_mask[j] & ~block_mask != 0:
                    new_uncovered.add(j)
            uncovered_set = new_uncovered
            
        block.sort()
        blocks.append(block)
        
    # Reduction Phase
    # Compute cover counts
    cover_count = [0] * num_tsubs
    block_masks = []
    for b in blocks:
        mask = 0
        for p in b:
            mask |= (1 << p)
        block_masks.append(mask)
        
    for i, mask in enumerate(block_masks):
        for j in range(num_tsubs):
            if tsub_point_mask[j] & ~mask == 0:
                cover_count[j] += 1
                
    # Iteratively remove blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            mask = block_masks[i]
            # Check if any t-subset is uniquely covered by this block
            unique = False
            for j in range(num_tsubs):
                if tsub_point_mask[j] & ~mask == 0:
                    if cover_count[j] == 1:
                        unique = True
                        break
            if not unique:
                # Remove block i
                for j in range(num_tsubs):
                    if tsub_point_mask[j] & ~mask == 0:
                        cover_count[j] -= 1
                blocks.pop(i)
                block_masks.pop(i)
                changed = True
                break # Restart loop as indices changed
                
    return blocks
