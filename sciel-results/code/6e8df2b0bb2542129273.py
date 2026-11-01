import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    # Precomputation
    all_points = list(range(v))
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_t_subsets = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all (t-1)-subsets for gain calculation
    all_tm1_subsets = list(itertools.combinations(all_points, t-1))
    n_tm1_subsets = len(all_tm1_subsets)
    tm1_subset_to_idx = {s: i for i, s in enumerate(all_tm1_subsets)}
    
    # Precompute gain table: for each (t-1)-subset and each point x, 
    # what is the ID of the t-subset formed by adding x?
    # gain_table[r_idx, x] = id or -1
    gain_table = np.full((n_tm1_subsets, v), -1, dtype=int)
    for r_idx, r in enumerate(all_tm1_subsets):
        r_set = set(r)
        for x in all_points:
            if x not in r_set:
                t_sub = tuple(sorted(r + (x,)))
                gain_table[r_idx, x] = subset_to_id[t_sub]
    
    def get_block_subsets(block_tuple):
        """Returns list of IDs of t-subsets contained in the block."""
        return [subset_to_id[t_sub] for t_sub in itertools.combinations(block_tuple, t)]

    def greedy_covering_optimized(random_state):
        uncovered = np.ones(n_t_subsets, dtype=bool)
        blocks = []
        uncovered_indices = np.arange(n_t_subsets)
        
        while len(uncovered_indices) > 0:
            # Pick random uncovered subset
            idx = random_state.choice(len(uncovered_indices))
            target_id = uncovered_indices[idx]
            target_subset = all_t_subsets[target_id]
            
            block = list(target_subset)
            
            while len(block) < k:
                gains = np.zeros(v, dtype=int)
                
                # Get (t-1)-subsets of current block
                for r in itertools.combinations(block, t-1):
                    r_idx = tm1_subset_to_idx[r]
                    ids = gain_table[r_idx]
                    safe_ids = np.where(ids == -1, 0, ids)
                    counts = uncovered[safe_ids].astype(int)
                    counts[ids == -1] = 0
                    gains += counts
                    
                # Exclude points already in block
                for x in block:
                    gains[x] = -1
                    
                # Pick x with max gain
                max_gain = np.max(gains)
                if max_gain <= 0:
                    valid_xs = [x for x in range(v) if x not in block]
                    x = random_state.choice(valid_xs)
                else:
                    max_indices = np.where(gains == max_gain)[0]
                    x = random_state.choice(max_indices.tolist())
                    
                block.append(x)
                
            block.sort()
            block_tuple = tuple(block)
            blocks.append(block_tuple)
            
            # Update uncovered
            block_ids = get_block_subsets(block_tuple)
            uncovered[block_ids] = False
            
            # Update uncovered_indices
            uncovered_indices = np.where(uncovered)[0]
            
        return [list(b) for b in blocks]

    def remove_redundant(blocks):
        if not blocks:
            return blocks
            
        # Count covers
        cover_count = np.zeros(n_t_subsets, dtype=int)
        block_ids_list = []
        for b in blocks:
            ids = get_block_subsets(tuple(b))
            block_ids_list.append(ids)
            cover_count[ids] += 1
            
        # Try to remove in multiple passes
        current_blocks = blocks
        current_ids = block_ids_list
        current_cover_count = cover_count
        
        for _ in range(5):
            new_blocks = []
            new_ids = []
            for i, b in enumerate(current_blocks):
                ids = current_ids[i]
                if np.min(current_cover_count[ids]) > 1:
                    # Remove
                    current_cover_count[ids] -= 1
                else:
                    # Keep
                    new_blocks.append(b)
                    new_ids.append(ids)
            
            if len(new_blocks) == len(current_blocks):
                break
            current_blocks = new_blocks
            current_ids = new_ids
            
        return current_blocks

    # Main loop with restarts
    best_blocks = None
    start_time = time.time()
    deadline = start_time + 110
    seed = 0
    
    while time.time() < deadline:
        random_state = random.Random(seed)
        seed += 1
        
        blocks = greedy_covering_optimized(random_state)
        blocks = remove_redundant(blocks)
        
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            
    return best_blocks
