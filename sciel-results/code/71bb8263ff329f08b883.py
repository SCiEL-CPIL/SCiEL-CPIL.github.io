import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # 1. Precompute t-subsets and indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_index = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t = len(all_t_subs)
    
    # Helper to get mask of t-subsets in a set `s` that contain a specific point `x`
    # s is a tuple/list of points, x is a point not in s
    # We need t-subsets of s+{x} that contain x.
    # This is equivalent to choosing t-1 points from s, and adding x.
    def get_mask_containing_x(s, x):
        mask = 0
        # s is a list of points. We need combinations of size t-1 from s.
        for sub in itertools.combinations(s, t - 1):
            # sub is a tuple of t-1 points from s
            # The full t-subset is sub + (x,)
            # Sort to match t_index key format (tuples are sorted if generated from sorted s? 
            # itertools.combinations preserves input order. 
            # t_index keys are sorted tuples from itertools.combinations(range(v), t).
            # So we must sort sub + (x,) to look up in t_index.
            full_sub = tuple(sorted(sub + (x,)))
            mask |= 1 << t_index[full_sub]
        return mask

    # Helper to get mask of all t-subsets in a set `s`
    def get_mask_all(s):
        mask = 0
        for sub in itertools.combinations(s, t):
            mask |= 1 << t_index[sub]
        return mask

    best_blocks = None
    best_count = float('inf')
    
    # 2. Multiple Greedy Runs
    num_runs = 50 # Can adjust based on speed
    
    for run in range(num_runs):
        if time.time() > deadline - 10:
            break
            
        # Random seed for this run
        random.seed(run * 12345 + 67890)
        
        # Initialize uncovered mask
        uncovered = (1 << num_t) - 1
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 5:
                break
                
            # Pick a random uncovered t-subset to start the block
            # Find an index where bit is 1
            # Efficiently finding a set bit in a large int:
            # We can iterate through indices? No, 11628 is small enough.
            # Or use bit operations.
            # Let's just pick a random index and check? 
            # Better: Maintain a list of uncovered indices? 
            # Since we update `uncovered` mask, maintaining a list is costly.
            # Let's just find the first set bit? 
            # Or random sample.
            
            # To ensure we cover the "hardest" parts, picking the first uncovered is a standard greedy heuristic.
            # But randomization helps escape local optima.
            # Let's pick a random uncovered subset.
            
            # Find any set bit
            # u = uncovered
            # idx = (u & -u).bit_length() - 1  # This gives the lowest set bit.
            # For randomization, we can shift right by a random amount?
            
            # Simple approach: Iterate through a shuffled list of indices? 
            # Too slow to shuffle 11k list every step.
            
            # Let's stick to "First Uncovered" for the base greedy, 
            # but randomize the TIE-BREAKING in point selection.
            
            # Find the first uncovered t-subset
            # We need to map bit index back to subset.
            # We have t_index (subset -> idx). We need idx -> subset.
            # Let's create inverse map.
            if not hasattr(construct, 'inv_t_index'):
                construct.inv_t_index = [None] * num_t
                for sub, idx in t_index.items():
                    construct.inv_t_index[idx] = sub
            
            # Find lowest set bit
            low_bit = uncovered & -uncovered
            start_idx = low_bit.bit_length() - 1
            start_sub = construct.inv_t_index[start_idx]
            
            # Start block with this subset
            block = list(start_sub)
            
            # Extend block to size k
            while len(block) < k:
                best_gain = -1
                best_x = None
                candidates = [x for x in range(v) if x not in block]
                
                # Shuffle candidates to break ties randomly
                random.shuffle(candidates)
                
                for x in candidates:
                    # Calculate gain
                    # Gain is number of NEWLY covered t-subsets
                    # These are t-subsets in block+{x} that contain x and are currently uncovered.
                    gain_mask = get_mask_containing_x(block, x)
                    gain = (uncovered & gain_mask).bit_count()
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        # If gain is 0, no point in adding? 
                        # But we must reach size k. 
                        # If all gains are 0, we are stuck? 
                        # No, we just fill with arbitrary points.
                        # But greedy usually finds gain > 0.
                        
                if best_x is None:
                    # Should not happen if v > len(block)
                    break
                    
                block.append(best_x)
            
            # Sort block for consistent representation
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered mask
            # Remove all t-subsets in block
            block_mask = get_mask_all(block)
            uncovered &= ~block_mask
            
        # 3. Reduction Phase
        if not blocks:
            continue
            
        # Count coverage for each t-subset
        counts = [0] * num_t
        for b in blocks:
            # b is a tuple
            for sub in itertools.combinations(b, t):
                counts[t_index[sub]] += 1
        
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                redundant = True
                for sub in itertools.combinations(b, t):
                    if counts[t_index[sub]] == 1:
                        redundant = False
                        break
                if redundant:
                    for sub in itertools.combinations(b, t):
                        counts[t_index[sub]] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
    if best_blocks is None:
        # Fallback: Return empty or a trivial cover? 
        # Must return valid. 
        # If we failed, return a large valid cover constructed greedily without time check?
        # For safety, just return the last incomplete one? No, must be valid.
        # Let's do one final greedy without time limit if we have time.
        # But if we are here, we likely ran out of time.
        # Return empty list? Invalid.
        # Let's assume the loop ran at least once.
        pass

    return [list(b) for b in best_blocks]
