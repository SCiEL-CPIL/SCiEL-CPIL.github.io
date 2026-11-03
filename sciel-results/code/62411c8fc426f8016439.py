import itertools
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    masks = [0] * n_blocks
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # Helper to run greedy with randomization
    def run_greedy(seed):
        random.seed(seed)
        uncovered = all_mask
        selected_indices = []
        used_blocks = [False] * n_blocks
        
        while uncovered:
            best_gain = -1
            candidates = []
            
            for i in range(n_blocks):
                if used_blocks[i]:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if not candidates:
                break
                
            # Random tie-breaking
            best_idx = random.choice(candidates)
            selected_indices.append(best_idx)
            used_blocks[best_idx] = True
            uncovered &= ~masks[best_idx]
            
        return selected_indices

    # Helper for Local Search: Redundancy Removal + Swap
    def local_search(selected_indices):
        # Convert to set for O(1) lookup
        selected_set = set(selected_indices)
        # Maintain a count of coverage for each t-subset? 
        # Bitmasks are easier for "is it covered" but harder for "is it uniquely covered".
        # For redundancy: 
        # A block i is redundant if for every t-subset in mask[i], 
        # there is another block j in selected_set \ {i} that covers it.
        # Equivalently: mask[i] & ~(OR of all other masks) == 0.
        
        # To speed up, we can compute the "union of others" quickly?
        # No, N is ~300. O(N^2) bitmask ops is fine.
        
        improved = True
        while improved:
            improved = False
            
            # Pass 1: Remove redundant blocks
            for idx_in_list in range(len(selected_indices) - 1, -1, -1):
                block_idx = selected_indices[idx_in_list]
                block_mask = masks[block_idx]
                
                # Check if block is redundant
                # We need to know if any t-subset in block_mask is ONLY covered by this block.
                # If yes, not redundant.
                # If no, redundant.
                
                # Efficient check:
                # Union of all other blocks
                union_others = 0
                for other_idx in selected_indices:
                    if other_idx != block_idx:
                        union_others |= masks[other_idx]
                
                if (block_mask & ~union_others) == 0:
                    # Redundant
                    selected_indices.pop(idx_in_list)
                    selected_set.remove(block_idx)
                    improved = True
                    break # Restart loop as structure changed
                    
            # Pass 2: Try Swaps (Remove i, Add j if j covers what i uniquely covered)
            # This is more complex. 
            # Let's try a simpler "Hill Climbing" move:
            # For each block i in solution:
            #   Remove i.
            #   Let U = uncovered sets.
            #   If U is empty, we already handled it in Pass 1.
            #   Find best block j not in solution that maximizes coverage of U.
            #   If j covers ALL of U, swap i with j.
            #   This keeps count same, but might allow Pass 1 to remove a block in next iteration.
            
            if len(selected_indices) > 0:
                for i in range(len(selected_indices)):
                    block_idx = selected_indices[i]
                    
                    # Temporarily remove
                    # Calculate uncovered if we remove block_idx
                    # This requires recalculating union of others.
                    # We can compute union_others once per i?
                    
                    # Actually, let's just try to find a replacement for each block.
                    # This is O(N_solution * N_candidates).
                    # N_solution ~ 300, N_candidates ~ 75000.
                    # 300 * 75000 = 22.5 million ops. 
                    # Each op is bit_count. 
                    # In Python, this might be too slow for 120s if done repeatedly.
                    
                    # Optimization: Only try swaps if we are stuck in redundancy removal.
                    # And limit the number of swap attempts per local search run.
                    
                    pass # Skipping complex swap for now to ensure speed. 
                    # The "Remove and Repair" is powerful enough usually.
                    
        return selected_indices

    # Main Logic: Run multiple greedy instances, keep best
    best_blocks = None
    best_count = float('inf')
    
    # Run a few greedy instances with different seeds
    # How many? 120s limit. 
    # One greedy run: 75k blocks * ~300 iterations? 
    # No, greedy picks ~300 blocks. 
    # Each step scans 75k blocks. 300 * 75k = 22.5M bit_counts.
    # Python bit_count is fast, but 22.5M might take ~2-5 seconds.
    # So we can run ~10-20 instances.
    
    import time
    start_time = time.time()
    max_time = 110 # Leave 10s margin
    
    seed_val = 42
    while time.time() - start_time < max_time:
        indices = run_greedy(seed_val)
        # Local Search
        indices = local_search(indices)
        
        if len(indices) < best_count:
            best_count = len(indices)
            best_blocks = indices
            
        seed_val += 1

    if best_blocks is None:
        # Fallback: single run
        indices = run_greedy(0)
        indices = local_search(indices)
        best_blocks = indices

    return [list(k_subsets[i]) for i in best_blocks]
