import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute all t-subsets as bitmasks
    all_t_subs = []
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_t_subs.append(mask)
    
    # Set of all t-subsets
    all_t_subs_set = set(all_t_subs)
    
    best_blocks = None
    best_len = float('inf')
    
    def time_ok():
        return time.time() < deadline

    attempt = 0
    # Estimate max attempts
    # If each attempt takes ~0.5s, we can do 200 attempts.
    # Let's cap at 500 attempts to be safe against slow environments.
    max_attempts = 500
    
    while attempt < max_attempts and time_ok():
        attempt += 1
        uncovered = set(all_t_subs_set)
        blocks = []
        
        # Greedy Phase
        while uncovered and time_ok():
            # Pick an uncovered subset
            first_mask = min(uncovered)
            
            # Initialize block with points from first_mask
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
            while len(block) < k and time_ok():
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
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for sub_mask in t_minus_1_subs:
                        cand_mask = sub_mask | (1 << x)
                        if cand_mask in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_ties = [x]
                    elif gain == best_gain:
                        best_ties.append(x)
                
                if best_x == -1:
                    # No gain, fill remaining with smallest indices
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
            
            # Update uncovered
            for comb in itertools.combinations(block, t):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                if mask in uncovered:
                    uncovered.discard(mask)
        
        if not uncovered:
            # Valid covering found
            # Redundancy Removal
            cover_count = {mask: 0 for mask in all_t_subs_set}
            
            for b in blocks:
                for comb in itertools.combinations(b, t):
                    mask = 0
                    for p in comb:
                        mask |= (1 << p)
                    cover_count[mask] += 1
            
            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    
                    b = blocks[i]
                    redundant = True
                    affected_masks = []
                    for comb in itertools.combinations(b, t):
                        mask = 0
                        for p in comb:
                            mask |= (1 << p)
                        if cover_count[mask] <= 1:
                            redundant = False
                            break
                        affected_masks.append(mask)
                    
                    if redundant:
                        for mask in affected_masks:
                            cover_count[mask] -= 1
                        blocks.pop(i)
                        changed = True
                        break
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [b[:] for b in blocks]
                
            # Early termination if we hit lower bound or close to it
            # LB = ceil(C(19,5)/C(8,5)) = 208
            if best_len <= 210:
                break
        else:
            # Incomplete
            pass

    if best_blocks is None:
        return []
        
    return best_blocks
