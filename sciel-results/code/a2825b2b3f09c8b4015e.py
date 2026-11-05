import itertools
import random
import time

def construct(v, k, t):
    all_points = range(v)
    
    def bits_to_mask(bits):
        m = 0
        for b in bits:
            m |= (1 << b)
        return m

    # Generate all t-subset masks
    all_t_masks = []
    for combo in itertools.combinations(all_points, t):
        all_t_masks.append(bits_to_mask(combo))
        
    def greedy_cover(seed):
        random.seed(seed)
        # Initialize uncovered
        uncovered_list = all_t_masks[:]
        uncovered_set = set(uncovered_list)
        blocks = []
        
        def mask_to_points(mask):
            return [i for i in range(v) if (mask >> i) & 1]
            
        while uncovered_list:
            # Pick random index
            idx = random.randint(0, len(uncovered_list) - 1)
            start_sub = uncovered_list[idx]
            
            # Remove start_sub from structures? 
            # No, it will be removed when the block is finalized.
            # But if we pick it, it's still in the list.
            # We just need to know it's uncovered.
            
            block_mask = start_sub
            block_pts = mask_to_points(block_mask)
            
            while len(block_pts) < k:
                best_x = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block_pts]
                
                # Pre-compute (t-1) subsets of current block
                t_minus_1_subs = list(itertools.combinations(block_pts, t-1))
                
                for x in candidates:
                    gain = 0
                    for sub in t_minus_1_subs:
                        m = 0
                        for p in sub:
                            m |= (1 << p)
                        m |= (1 << x)
                        if m in uncovered_set:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and gain > 0:
                        if random.random() < 0.5:
                            best_x = x
                            best_gain = gain
                            
                if best_x == -1:
                    best_x = candidates[0]
                    
                block_pts.append(best_x)
                block_mask |= (1 << best_x)
                
            # Finalize block
            blocks.append(sorted(block_pts))
            
            # Remove all t-subsets of this block from uncovered structures
            for sub in itertools.combinations(block_pts, t):
                m = bits_to_mask(sub)
                if m in uncovered_set:
                    # Remove from set
                    uncovered_set.remove(m)
                    # Remove from list: find index? 
                    # This is O(N). 
                    # To make it O(1), we need a map from mask to index in list.
                    # Or just use a set for membership and a list for random pick, 
                    # and periodically rebuild the list?
                    # Or simpler: just use the set for everything and accept O(N) pick?
                    # Given the low number of iterations, O(N) pick is likely fine.
                    pass
                    
        return blocks

    # Re-evaluating the list removal.
    # If we just use a set `uncovered`, `random.choice(list(uncovered))` is O(N).
    # N=11628. 300 blocks. 3.5M operations. 
    # In Python, 3.5M simple ops is ~0.1s.
    # The gain calculation is 630k ops.
    # Total time per run ~0.2s.
    # 120s / 0.2s = 600 runs.
    # This is plenty.
    
    # Let's stick to the simpler Set-based implementation to avoid bugs in list management.
    
    def greedy_cover_simple(seed):
        random.seed(seed)
        uncovered = set(all_t_masks)
        blocks = []
        
        def mask_to_points(mask):
            return [i for i in range(v) if (mask >> i) & 1]
            
        while uncovered:
            # Random pick
            # Convert to list to allow random choice
            # Optimization: We can pick a random element from a set by converting to list.
            # Or we can just pick the min? No, random is better for restarts.
            start_sub = random.choice(list(uncovered))
            block_mask = start_sub
            block_pts = mask_to_points(block_mask)
            
            while len(block_pts) < k:
                best_x = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block_pts]
                
                t_minus_1_subs = list(itertools.combinations(block_pts, t-1))
                
                for x in candidates:
                    gain = 0
                    for sub in t_minus_1_subs:
                        m = 0
                        for p in sub:
                            m |= (1 << p)
                        m |= (1 << x)
                        if m in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and gain > 0:
                        if random.random() < 0.5:
                            best_x = x
                            best_gain = gain
                            
                if best_x == -1:
                    best_x = candidates[0]
                    
                block_pts.append(best_x)
                block_mask |= (1 << best_x)
                
            blocks.append(sorted(block_pts))
            
            for sub in itertools.combinations(block_pts, t):
                m = bits_to_mask(sub)
                if m in uncovered:
                    uncovered.remove(m)
                    
        return blocks

    best_blocks = None
    start_time = time.time()
    timeout = 110
    
    seed = 42
    while time.time() - start_time < timeout:
        blocks = greedy_cover_simple(seed)
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
        seed += 1
        
    return best_blocks
