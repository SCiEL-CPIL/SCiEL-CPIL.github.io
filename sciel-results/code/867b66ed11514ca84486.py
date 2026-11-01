import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    tset_masks = [0] * len(all_tsets)
    mask_to_idx = {}
    for i, comb in enumerate(all_tsets):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        tset_masks[i] = mask
        mask_to_idx[mask] = i
        
    num_tsets = len(all_tsets)
    
    best_blocks = None
    best_len = float('inf')
    
    # Run greedy with different tie-breakers/randomizations
    # Since the problem is small, we can try a few variations
    
    for attempt in range(10): # Try 10 times
        if time.time() > deadline:
            break
            
        uncovered = np.ones(num_tsets, dtype=bool)
        uncovered_indices = set(range(num_tsets))
        blocks = []
        
        while uncovered_indices:
            if time.time() > deadline:
                break
                
            # Pick a start t-set
            # Randomly pick one to diversify
            start_idx = next(iter(uncovered_indices)) # Or random choice
            # To be deterministic but diverse, we can shuffle the order of selection?
            # Let's stick to picking the first one, but randomize the candidate order in gain calculation?
            # Or just pick a random uncovered t-set.
            start_idx = list(uncovered_indices)[attempt % len(uncovered_indices)] # Simple variation
            
            start_mask = tset_masks[start_idx]
            current_points = [p for p in range(v) if start_mask & (1 << p)]
            
            while len(current_points) < k:
                best_x = -1
                best_gain = -1
                
                # Randomize candidate order for tie-breaking diversity
                candidates = [x for x in range(v) if not (start_mask & (1 << x) or (len(current_points)>0 and any(p==x for p in current_points[1:])))]
                # Simpler:
                current_mask = 0
                for p in current_points:
                    current_mask |= (1 << p)
                
                candidates = [x for x in range(v) if not (current_mask & (1 << x))]
                
                if attempt % 2 == 0:
                    candidates.reverse() # Different order for different attempts?
                
                for x in candidates:
                    new_points = current_points + [x]
                    # Calculate gain
                    gain = 0
                    # Generate 5-subsets of new_points
                    # Optimization: Only check subsets that include the new point x?
                    # No, the whole set is covered. But we only care about NEW coverage.
                    # Actually, any t-set in new_points that is in uncovered is a gain.
                    # Since previous points were already in current_points, any t-set NOT containing x was already considered in previous steps?
                    # No, a t-set containing x is new. A t-set not containing x was a subset of current_points.
                    # If it was uncovered, it would have been covered by previous blocks?
                    # Not necessarily.
                    # So we must check all 5-subsets of new_points.
                    
                    for sub in itertools.combinations(new_points, t):
                        sub_mask = 0
                        for p in sub:
                            sub_mask |= (1 << p)
                        idx = mask_to_idx[sub_mask]
                        if uncovered[idx]:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x == -1:
                    # Should not happen if uncovered is not empty
                    break
                    
                current_points.append(best_x)
                
            # Finalize block
            final_points = current_points
            final_mask = 0
            for p in final_points:
                final_mask |= (1 << p)
                
            # Update uncovered
            for sub in itertools.combinations(final_points, t):
                sub_mask = 0
                for p in sub:
                    sub_mask |= (1 << p)
                idx = mask_to_idx[sub_mask]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_indices.remove(idx)
                    
            blocks.append(final_points)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
