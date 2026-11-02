import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Precompute all t-subsets and map to index
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # covered[i] = 1 if subset i is covered
    covered = bytearray(n_t)
    uncovered_count = n_t
    
    blocks = []
    
    # Precompute combinations of size t-1 for each point?
    # No, we compute gain on the fly.
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        # Find first uncovered subset
        # Optimization: maintain a list of uncovered indices?
        # Or just scan. n_t is ~11k, scanning is fast.
        first_uncovered_idx = -1
        for i in range(n_t):
            if covered[i] == 0:
                first_uncovered_idx = i
                break
        
        if first_uncovered_idx == -1:
            break
            
        # Start block with this subset
        block = list(all_t_subsets[first_uncovered_idx])
        
        # Greedily add points
        while len(block) < k:
            best_gain = -1
            best_point = -1
            
            # Candidates
            block_set = set(block)
            candidates = [x for x in range(v) if x not in block_set]
            
            # Precompute (t-1)-subsets of current block
            # For gain calculation: adding x covers subsets {x} U C where C in C(block, t-1)
            block_t_minus_1 = list(itertools.combinations(block, t-1))
            
            for x in candidates:
                gain = 0
                for C in block_t_minus_1:
                    # Form new subset
                    # C is tuple, x is int
                    # Need to sort to match subset_to_idx key
                    # Optimization: C is sorted. x is not necessarily.
                    # Merge x into C
                    # Since C is sorted, we can find insertion point
                    # But t is small (5), so just sort tuple(C + (x,))
                    new_sub = tuple(sorted(C + (x,)))
                    idx = subset_to_idx[new_sub]
                    if covered[idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point != -1:
                block.append(best_point)
            else:
                # Should not happen if uncovered_count > 0
                # But if all gains are 0, we still need to add a point to form a valid block?
                # If gain is 0, adding x doesn't cover anything new.
                # We must add a point to reach size k.
                # Pick any candidate.
                block.append(candidates[0])
        
        block.sort()
        blocks.append(block)
        
        # Update covered
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            if covered[idx] == 0:
                covered[idx] = 1
                uncovered_count -= 1
    
    # Delete redundant blocks
    if time.time() < deadline - 5: # Leave 5s for finishing
        # Calculate cover counts
        cover_count = [0] * n_t
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                cover_count[idx] += 1
        
        changed = True
        while changed:
            if time.time() > deadline - 2:
                break
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                block = blocks[i]
                redundant = True
                for sub in itertools.combinations(block, t):
                    idx = subset_to_idx[sub]
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    for sub in itertools.combinations(block, t):
                        idx = subset_to_idx[sub]
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart loop as indices changed
    
    return blocks
