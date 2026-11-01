import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    # Precompute all t-subsets and map to IDs
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    
    # covered[i] is True if subset i is covered
    covered = np.zeros(num_subsets, dtype=bool)
    # cover_counts[i] is number of blocks covering subset i
    cover_counts = np.zeros(num_subsets, dtype=np.int32)
    
    blocks = []
    
    # Helper to update covered and cover_counts for a block
    def add_block(block):
        block_set = set(block)
        for s in itertools.combinations(block, t):
            idx = subset_to_id[s]
            if not covered[idx]:
                covered[idx] = True
            cover_counts[idx] += 1
        blocks.append(block)

    # Helper to remove a block
    def remove_block(block_idx):
        block = blocks[block_idx]
        for s in itertools.combinations(block, t):
            idx = subset_to_id[s]
            cover_counts[idx] -= 1
            if cover_counts[idx] == 0:
                covered[idx] = False
        blocks.pop(block_idx)

    # Greedy Construction
    while covered.sum() < num_subsets:
        if time.time() - start_time > timeout:
            break
            
        # Find an uncovered subset
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        # Pick the first one (or random, or most constrained)
        # First one is fine for now
        target_subset = all_subsets[uncovered_indices[0]]
        
        current_block = list(target_subset)
        
        while len(current_block) < k:
            if time.time() - start_time > timeout:
                break
                
            best_point = -1
            best_gain = -1
            
            # Candidates are points not in current_block
            candidates = [x for x in range(v) if x not in current_block]
            
            # Precompute combinations of current_block of size t-1
            # To check if adding x covers new subsets
            # A subset s is covered by adding x if s contains x and s\{x} is subset of current_block
            # Since |s|=t, s\{x} has size t-1.
            # So we look at all y in C(current_block, t-1). s = y U {x}.
            
            block_combos = list(itertools.combinations(current_block, t-1))
            
            for x in candidates:
                gain = 0
                for y in block_combos:
                    s = tuple(sorted(y + (x,)))
                    idx = subset_to_id[s]
                    if not covered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point != -1:
                current_block.append(best_point)
            else:
                # Should not happen if valid parameters
                break
        
        if len(current_block) == k:
            current_block.sort()
            add_block(current_block)
        
        if time.time() - start_time > timeout:
            break

    # Local Search: Remove redundant blocks
    # We can do multiple passes
    progress = True
    while progress:
        if time.time() - start_time > timeout:
            break
        progress = False
        # Iterate from last to first? Or random?
        # Let's iterate from 0 to end, but if we remove, indices shift.
        # Better: iterate in reverse
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() - start_time > timeout:
                break
            block = blocks[i]
            # Check if all subsets in this block are covered by other blocks
            # i.e., cover_counts[s] > 1 for all s in C(block, t)
            redundant = True
            for s in itertools.combinations(block, t):
                idx = subset_to_id[s]
                if cover_counts[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                remove_block(i)
                progress = True
    
    return blocks
