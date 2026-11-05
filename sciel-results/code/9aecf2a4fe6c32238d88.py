"""Covering design C(v,k,t) via greedy set cover with on-the-fly candidate generation."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    covered = np.zeros(n_t, dtype=np.uint8)
    uncovered_count = n_t
    blocks = []
    
    # Greedy Phase
    while uncovered_count > 0 and time_ok():
        # Find an uncovered t-subset to anchor the search
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Pick a random uncovered t-subset
        anchor_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        anchor_t_sub = all_t_subs[anchor_idx]
        anchor_set = set(anchor_t_sub)
        
        # Generate all k-blocks containing anchor_t_sub
        remaining_points = [p for p in all_points if p not in anchor_set]
        k_minus_t = k - t
        
        best_block = None
        best_gain = -1
        
        # Iterate through all combinations of (k-t) points from remaining_points
        for ext in itertools.combinations(remaining_points, k_minus_t):
            # Ensure the block is sorted so that itertools.combinations yields sorted tuples
            # which match the keys in t_sub_to_idx
            block = sorted(list(anchor_t_sub) + list(ext))
            # Calculate gain: how many uncovered t-subsets are in this block?
            gain = 0
            for t_sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[t_sub]
                if covered[idx] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            break
            
        # Add best_block to our solution
        blocks.append(best_block)
        
        # Update coverage
        for t_sub in itertools.combinations(best_block, t):
            idx = t_sub_to_idx[t_sub]
            if covered[idx] == 0:
                covered[idx] = 1
                uncovered_count -= 1
                
    # Reduction Phase: Remove redundant blocks
    if blocks:
        # Precompute t-subsets for each block to speed up reduction
        block_t_subs = []
        for b in blocks:
            subs = [t_sub_to_idx[ts] for ts in itertools.combinations(b, t)]
            block_t_subs.append(subs)
            
        cover_count = np.zeros(n_t, dtype=np.int32)
        for subs in block_t_subs:
            cover_count[subs] += 1
        
        changed = True
        while changed and time_ok():
            changed = False
            new_blocks = []
            new_block_t_subs = []
            for i, subs in enumerate(block_t_subs):
                # Check if any t-subset in this block has cover_count == 1
                keep = False
                for idx in subs:
                    if cover_count[idx] == 1:
                        keep = True
                        break
                if keep:
                    new_blocks.append(blocks[i])
                    new_block_t_subs.append(subs)
                else:
                    # Remove block
                    changed = True
                    cover_count[subs] -= 1
            
            blocks = new_blocks
            block_t_subs = new_block_t_subs
    
    return blocks
# EVOLVE-BLOCK-END
