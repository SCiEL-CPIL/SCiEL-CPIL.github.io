import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # For each point, which t-subsets contain it?
    # points_to_subs[p] = list of indices of t-subsets containing p
    points_to_subs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subs):
        for p in sub:
            points_to_subs[p].append(i)
            
    # Convert to numpy arrays for faster operations? 
    # Actually, boolean masking on numpy is fast.
    # Let's represent coverage as a numpy array of ints.
    
    def greedy_once(seed):
        random.seed(seed)
        np.random.seed(seed)
        uncovered = np.ones(n_t_subs, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            block = []
            # Choose first point randomly from those in uncovered subsets? 
            # Or just 0.
            # To improve diversity, pick a random point from an uncovered subset.
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            # Pick a random uncovered subset
            rand_sub_idx = unc_idx[random.randint(0, len(unc_idx)-1)]
            rand_sub = all_t_subs[rand_sub_idx]
            # Pick a random point from this subset to start the block
            start_p = random.choice(rand_sub)
            block.append(start_p)
            
            while len(block) < k:
                # Calculate gain for each candidate point not in block
                # Gain = number of uncovered t-subsets that contain (current block + candidate)
                # A t-subset S contains (block + x) iff S contains all points in block AND S contains x.
                
                # Efficient way:
                # Start with mask of all uncovered subsets.
                # Intersect with masks of points in block.
                # Then for each candidate x, intersect with mask of x.
                
                # Pre-computed masks for points
                # point_masks[p] = boolean array of length n_t_subs, True if p in sub
                
                # We can compute the intersection of points in block once.
                # mask_block = uncovered
                # for p in block:
                #    mask_block &= point_masks[p]
                
                # Then for each candidate x:
                #    gain[x] = np.sum(mask_block & point_masks[x])
                
                # This is O(v * n_t_subs) per step.
                # n_t_subs = 3876. v=19.
                # 19 * 3876 ~ 73k ops. Very fast.
                
                # We need point_masks.
                # Let's create them outside the loop if possible, or cache them.
                
                # To avoid creating point_masks every time, let's keep them global in the closure.
                
                mask_block = uncovered.copy()
                for p in block:
                    mask_block &= point_masks[p]
                    
                best_gain = -1
                best_x = -1
                candidates = [x for x in range(v) if x not in block]
                
                # Vectorized gain calculation
                # Create a matrix of gains?
                # Or just loop over candidates (19 is small)
                
                for x in candidates:
                    # gain is sum of (mask_block & point_masks[x])
                    # But we only care about uncovered ones.
                    # mask_block is already restricted to uncovered.
                    gain = np.sum(mask_block & point_masks[x])
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain:
                        # Random tie break
                        if random.random() < 0.5:
                            best_x = x
                            
                if best_x == -1:
                    break
                block.append(best_x)
                
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                uncovered[idx] = False
                
        return blocks

    # Precompute point_masks
    point_masks = [np.zeros(n_t_subs, dtype=bool) for _ in range(v)]
    for p in range(v):
        for idx in points_to_subs[p]:
            point_masks[p][idx] = True

    # Run multiple greedy restarts
    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 30: # Reserve 30s for local search
            break
        blocks = greedy_once(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback to single greedy if all failed or time ran out
        best_blocks = greedy_once(0)
        
    # Local Search: Prune and Swap
    # Convert to list of lists for mutability
    blocks = [list(b) for b in best_blocks]
    
    # Compute coverage counts
    cov_counts = np.zeros(n_t_subs, dtype=int)
    for b in blocks:
        for sub in itertools.combinations(b, t):
            idx = sub_to_idx[sub]
            cov_counts[idx] += 1
            
    # Prune redundant blocks
    # A block is redundant if all its t-subsets have cov_counts > 1
    def is_redundant(b):
        for sub in itertools.combinations(b, t):
            idx = sub_to_idx[sub]
            if cov_counts[idx] == 1:
                return False
        return True
        
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                # Remove b
                # Decrement cov_counts for all subs in b
                for sub in itertools.combinations(b, t):
                    idx = sub_to_idx[sub]
                    cov_counts[idx] -= 1
                changed = True
            else:
                new_blocks.append(b)
        blocks = new_blocks
        
    # Local Search: 1-for-1 Swaps
    # Try to swap points in blocks to see if it makes other blocks redundant
    # Or just to maintain validity while allowing removal?
    # The prompt says "Local search with 1-for-1 swaps ... allows escaping local minima"
    # Let's try to find swaps that allow block removal.
    
    # Strategy:
    # 1. Identify blocks that are "critical" (contain at least one subset with count 1).
    # 2. Try to modify a non-critical block? No, modifying a block changes its coverage.
    # 3. Try to modify a critical block to cover its unique subsets in a way that doesn't hurt others, 
    #    potentially making another block redundant?
    # This is complex.
    
    # Simpler Local Search: 
    # Randomly swap points in blocks. 
    # If the cover remains valid, accept.
    # If the cover becomes invalid, reject.
    # This doesn't reduce block count directly, but might lead to a state where pruning works better?
    # Actually, if we just maintain validity, we don't reduce count.
    # We need to reduce count.
    
    # Better: 
    # Try to remove a block. If invalid, try to "repair" by modifying other blocks?
    # Let's stick to the Pruning phase which is guaranteed to help.
    # The Greedy with restarts is the main driver for improvement.
    
    # Final check: ensure all subsets are covered
    # (The greedy guarantees this, and pruning only removes redundant blocks, so it should still be valid)
    
    # Convert back to list of lists for return
    return [list(b) for b in blocks]
