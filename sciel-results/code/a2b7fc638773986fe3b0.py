import itertools
import random
import time

def construct(v, k, t):
    # The task specifies v=19, k=7, t=4.
    # We implement a specialized solver for this case for maximum performance.
    # If parameters differ, we fall back to a generic greedy.
    
    if v != 19 or k != 7 or t != 4:
        return _generic_greedy(v, k, t)

    # Precomputation for C(19, 7, 4)
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Bitmask for all subsets covered
    # We use Python integers as arbitrary precision bitmasks
    
    def get_block_mask(block):
        mask = 0
        for subset in itertools.combinations(block, t):
            idx = subset_to_idx[subset]
            mask |= (1 << idx)
        return mask

    def run_greedy(seed):
        random.seed(seed)
        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        
        while uncovered_mask:
            # Find an uncovered subset to start the block
            # Using the lowest set bit for determinism within the run's random state
            # (randomness comes from tie-breaking in expansion)
            lsb = uncovered_mask & -uncovered_mask
            anchor_idx = lsb.bit_length() - 1
            anchor_subset = all_t_subsets[anchor_idx]
            
            block = list(anchor_subset)
            
            # Expand the block to size k
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [p for p in points if p not in block]
                
                # Shuffle to break ties randomly
                random.shuffle(candidates)
                
                for p in candidates:
                    temp_block = block + [p]
                    # Calculate gain: number of currently uncovered subsets 
                    # that would be covered by adding p to the block
                    temp_mask = get_block_mask(temp_block)
                    gain = (temp_mask & uncovered_mask).bit_count()
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point == -1:
                    # Should not happen if v > k
                    break
                    
                block.append(best_point)
            
            # Sort block for canonical representation
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            block_mask = get_block_mask(block)
            uncovered_mask &= ~block_mask
        
        # Reduction Phase
        # Calculate coverage count for each t-subset
        cover_count = [0] * n_subsets
        for b in blocks:
            for subset in itertools.combinations(b, t):
                cover_count[subset_to_idx[subset]] += 1
        
        # Iteratively remove non-essential blocks
        changed = True
        while changed:
            changed = False
            # Iterate in reverse order, or random order?
            # Reverse order is deterministic. 
            # Let's try random order of indices to be more robust
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                b = blocks[i]
                essential = False
                for subset in itertools.combinations(b, t):
                    idx = subset_to_idx[subset]
                    if cover_count[idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove block i
                    for subset in itertools.combinations(b, t):
                        idx = subset_to_idx[subset]
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart the shuffle/iteration
        
        return blocks

    def _generic_greedy(v, k, t):
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    best_blocks = None
    best_len = float('inf')
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    seed = 0
    while True:
        if time.time() - start_time > time_limit:
            break
        
        blocks = run_greedy(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
        # Early termination if we hit the lower bound (131)
        if best_len <= 131:
            break

    if best_blocks is None:
        return []
        
    return best_blocks
