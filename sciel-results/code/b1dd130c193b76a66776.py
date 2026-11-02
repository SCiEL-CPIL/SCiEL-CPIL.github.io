import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Precomputation
    # Generate all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute bitmask for each block
    # Mask bit i is 1 if the block covers t_subset i
    block_masks = [0] * num_blocks
    # Also store list of t-subset indices for each block for fast elimination
    block_t_indices = [[] for _ in range(num_blocks)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        indices = []
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[b_idx] = mask
        block_t_indices[b_idx] = indices

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy instances with randomized tie-breaking
    # The number of runs depends on time.
    # Each run: Greedy (O(N_blocks * N_steps * bit_count)) + Elimination (O(N_blocks * N_steps * C(k,t)))
    
    max_runs = 50 # Start with a reasonable number, adjust if fast
    
    for run_idx in range(max_runs):
        if time.time() > deadline:
            break
            
        random.seed(run_idx)
        
        uncovered_mask = (1 << num_t_subsets) - 1
        selected_block_indices = []
        
        # Greedy Construction
        # To speed up finding the best block, we iterate all blocks.
        # Optimization: If we have a list of blocks, we can just loop.
        # 75k blocks * ~200 steps = 15 million iterations.
        # Each iteration: (mask & uncovered).bit_count()
        # This is likely fast enough in Python if optimized.
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_candidates = []
            
            # Find all blocks with max gain
            # To allow randomization, we collect top candidates
            # Or just find max and pick random from ties?
            # Finding all ties is expensive.
            # Alternative: Add small random noise to gain?
            # gain + random.uniform(0, 0.1)
            
            # Let's just find the max, and if there are ties, pick random.
            # But finding ties requires storing all maxes.
            # Let's just pick the first max, but shuffle the block order?
            # No, shuffling 75k list every step is expensive.
            
            # Better: Randomly perturb the gain slightly.
            # This effectively breaks ties randomly.
            
            max_gain = -1
            chosen_idx = -1
            
            # We can optimize by not checking blocks that are already selected?
            # No, a block can only be selected once. We should maintain a set of available blocks?
            # Or just mark selected ones and skip them.
            # Since we are building a set, we should not pick the same block twice.
            # Let's maintain a `used` boolean array or set.
            
            # Actually, picking the same block twice is useless.
            # Let's keep a set of used indices.
            
            # For speed, let's just iterate all and skip used.
            
            for b_idx in range(num_blocks):
                if b_idx in selected_block_indices: # O(N) check, bad. Use set.
                    continue
                # We need a set for O(1) lookup.
                # Let's maintain `used_blocks` as a set.
                
            # Re-implementing loop with set
            used_blocks = set(selected_block_indices)
            
            max_gain = -1
            candidates = []
            
            for b_idx in range(num_blocks):
                if b_idx in used_blocks:
                    continue
                
                gain = (block_masks[b_idx] & uncovered_mask).bit_count()
                
                # Add small random noise to break ties
                # This makes the search stochastic
                gain += random.random() * 0.01
                
                if gain > max_gain:
                    max_gain = gain
                    candidates = [b_idx]
                elif abs(gain - max_gain) < 1e-9:
                    candidates.append(b_idx)
                    
            if not candidates:
                break
                
            chosen_idx = random.choice(candidates)
            selected_block_indices.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            uncovered_mask &= ~block_masks[chosen_idx]
            
        # If we timed out during greedy, skip this run or use partial?
        # Partial is not a valid covering. Must be complete.
        # So if we broke due to time, we should check if uncovered_mask is 0.
        if uncovered_mask != 0:
            continue
            
        # Local Search: Elimination
        # Use coverage counts
        coverage_count = [0] * num_t_subsets
        for b_idx in selected_block_indices:
            for idx in block_t_indices[b_idx]:
                coverage_count[idx] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle to find different redundancies?
            # Elimination order can matter if we do swaps, but for pure elimination,
            # removing any redundant block is valid.
            # However, removing one might make another redundant?
            # No, removing a block only decreases coverage counts.
            # If a block was not redundant (had a unique subset), it remains not redundant
            # or becomes redundant?
            # If I remove block A, coverage counts decrease.
            # A block B was redundant if all its subsets had count > 1.
            # If I remove A, counts decrease. B might still be redundant.
            # A block C was not redundant (had a subset with count 1).
            # If that subset was in A, and I remove A, the count becomes 0.
            # So C is still not redundant (it's the only cover now).
            # So removal of redundant blocks is safe and order doesn't create new non-redundancies
            # for other blocks?
            # Wait. If I remove A, I might uncover a subset S.
            # If S was covered by A and B, count was 2. Now count is 1 (covered by B).
            # B was redundant (count 2 for all). Now B has a subset with count 1.
            # So B becomes NON-redundant.
            # So we must iterate until no more changes.
            
            # We can do this in one pass if we are careful, but multiple passes are safer.
            
            new_selected = []
            for b_idx in selected_block_indices:
                # Check if redundant
                redundant = True
                for idx in block_t_indices[b_idx]:
                    if coverage_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove
                    for idx in block_t_indices[b_idx]:
                        coverage_count[idx] -= 1
                    improved = True
                else:
                    new_selected.append(b_idx)
                    
            selected_block_indices = new_selected
            
        if len(selected_block_indices) < best_count:
            best_count = len(selected_block_indices)
            best_blocks = [list(all_blocks[i]) for i in selected_block_indices]
            
    if best_blocks is None:
        return []
        
    return best_blocks
