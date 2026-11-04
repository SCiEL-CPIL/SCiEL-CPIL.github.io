import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    # Precompute all t-subsets as bitmasks and map to indices
    all_t_subs = []
    mask_to_idx = {}
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_t_subs.append(mask)
        mask_to_idx[mask] = len(all_t_subs) - 1
    
    num_t_subs = len(all_t_subs)
    max_mask = 1 << v

    best_blocks = None
    best_len = float('inf')
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        
        # Use a boolean list for O(1) lookup, faster than set of ints
        is_uncovered = [True] * max_mask
        for mask in all_t_subs:
            is_uncovered[mask] = True
            
        blocks = []
        
        # Greedy Phase
        while time.time() < deadline:
            # Pick an uncovered subset
            # Scan all_t_subs for the first uncovered one
            first_mask = -1
            for mask in all_t_subs:
                if is_uncovered[mask]:
                    first_mask = mask
                    break
            
            if first_mask == -1:
                break # All covered
            
            # Extract points from mask
            init_points = []
            m = first_mask
            idx = 0
            while m:
                if m & 1:
                    init_points.append(idx)
                m >>= 1
                idx += 1
            
            block = init_points
            block_set = set(block)
            
            # Grow the block
            while len(block) < k and time.time() < deadline:
                best_x = -1
                best_gain = -1
                best_ties = []
                
                # Generate all (t-1) subsets of current block
                t_minus_1_subs = []
                for comb in itertools.combinations(block, t - 1):
                    mask = 0
                    for p in comb:
                        mask |= (1 << p)
                    t_minus_1_subs.append(mask)
                
                # Evaluate gain for each candidate x
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    x_bit = 1 << x
                    for sub_mask in t_minus_1_subs:
                        if is_uncovered[sub_mask | x_bit]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_ties = [x]
                    elif gain == best_gain:
                        best_ties.append(x)
                
                if best_x == -1:
                    # No gain, just fill remaining
                    for x in range(v):
                        if len(block) >= k:
                            break
                        if x not in block_set:
                            block.append(x)
                            block_set.add(x)
                    break
                
                # Random tie-breaking
                if len(best_ties) > 1:
                    best_x = random.choice(best_ties)
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update is_uncovered
            for comb in itertools.combinations(block, t):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                is_uncovered[mask] = False
        
        # Check if valid (first_mask was -1)
        # We can assume it is valid if the loop terminated naturally.
        # However, if time ran out, it might not be valid.
        # We should verify or just skip if not complete.
        # A simple check: if any is_uncovered is True, it's invalid.
        # But checking 500k items is fast.
        valid = True
        for mask in all_t_subs:
            if is_uncovered[mask]:
                valid = False
                break
        
        if valid:
            # Redundancy Removal
            cover_count = [0] * num_t_subs
            
            for b in blocks:
                for comb in itertools.combinations(b, t):
                    mask = 0
                    for p in comb:
                        mask |= (1 << p)
                    cover_count[mask_to_idx[mask]] += 1
            
            # Try to remove redundant blocks
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(blocks)):
                    if time.time() > deadline:
                        break
                    
                    b = blocks[i]
                    redundant = True
                    affected_indices = []
                    for comb in itertools.combinations(b, t):
                        mask = 0
                        for p in comb:
                            mask |= (1 << p)
                        idx = mask_to_idx[mask]
                        if cover_count[idx] <= 1:
                            redundant = False
                            break
                        affected_indices.append(idx)
                    
                    if redundant:
                        for idx in affected_indices:
                            cover_count[idx] -= 1
                        blocks.pop(i)
                        changed = True
                        break
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [b[:] for b in blocks]
                
            if best_len <= 210:
                break

        # Check time for next attempt
        if time.time() > deadline - 2.0:
            break

    if best_blocks is None:
        return []
        
    return best_blocks
