import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Precomputation
    points = range(v)
    all_3s = list(itertools.combinations(points, t))
    n_3s = len(all_3s)
    # Map 3-subsets to indices
    idx_map = {comb: i for i, comb in enumerate(all_3s)}
    
    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute the 3-subsets contained in each k-subset
    # block_3_indices[i] is an array of indices into all_3s
    block_3_indices = np.zeros((n_ks, C(k, t)), dtype=np.int32)
    # Helper to compute C(k, t)
    def nCr(n, r):
        return int(np.prod([n-i for i in range(r)]) / np.prod([i+1 for i in range(r)]))
    t_count = nCr(k, t)
    block_3_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_3_indices[i, j] = idx_map[sub]
            
    # Greedy Construction
    covered = np.zeros(n_3s, dtype=bool)
    blocks = []
    point_load = np.zeros(v, dtype=int)
    
    while True:
        if time.time() - start_time > time_limit * 0.5: # Use half time for greedy
            break
            
        # Calculate gains for all blocks
        # is_uncovered = ~covered
        # gains = sum over each block's 3s of is_uncovered[idx]
        # Vectorized:
        # block_3_indices shape (n_ks, t_count)
        # We need to index covered with these indices
        # This creates a (n_ks, t_count) boolean array
        # Then sum along axis 1
        
        # Optimization: Only compute for blocks that might be good?
        # For now, do all.
        try:
             uncovered_flags = ~covered[block_3_indices]
             gains = uncovered_flags.sum(axis=1)
        except Exception as e:
             # Fallback if memory or time issues
             break
             
        # Tie-breaking: Prefer blocks with lower point load sum
        # point_load shape (v,)
        # block_points shape (n_ks, k) - need to precompute or compute on fly?
        # Precomputing block points as indices is easy.
        
        # Let's add tie-breaking logic
        # Sum of point_load for each block
        # We need block_points array (n_ks, k)
        
        # Re-precompute block points if not done
        # Actually, we can just pick the first max gain if tie-breaking is complex.
        # But balance is important.
        
        # Let's just pick max gain.
        max_gain = np.max(gains)
        if max_gain == 0:
             break # All covered? Or no progress?
             
        # Find all blocks with max gain
        candidates = np.where(gains == max_gain)[0]
        
        # Tie break: min sum of point_load
        # To do this fast, we need block_points
        # Let's assume we precomputed block_points in the setup
        
        # For simplicity in this step, pick random from candidates to avoid bias
        # Or pick the one with min load.
        
        # Let's refine: Precompute block_points
        # block_points = np.array(all_ks)
        
        # sum_loads = block_points @ point_load
        # min_load_idx = np.argmin(sum_loads[candidates])
        # best_idx = candidates[min_load_idx]
        
        # To save space/time in this thought process, I will implement the full version in the code block.
        
        # Add block
        # ...
        
    # Local Search
    # ...
    
    return [list(b) for b in blocks]
