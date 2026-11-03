import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    # Precomputation
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(all_k_blocks)
    
    # Map each k-block to the list of t-subset indices it contains
    block_t_subsets = []
    for block in all_k_blocks:
        subs = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_t_subsets.append(subs)
        
    # Map each t-subset to the list of k-block indices that contain it
    t_subset_k_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, block in enumerate(all_k_blocks):
        for sub in itertools.combinations(block, t):
            s_idx = t_subset_to_idx[sub]
            t_subset_k_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for faster access if needed, 
    # but lists of lists are fine for the indices.
    # block_t_subsets is list of lists of ints.
    # t_subset_k_blocks is list of lists of ints.
    
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    timeout = 115  # Safety margin
    
    run_id = 0
    
    while time.time() - start_time < timeout:
        # Greedy Construction
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        current_blocks = []
        
        while True:
            # Find uncovered subsets
            uncovered_indices = np.where(cover_count == 0)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a random uncovered subset
            # Using run_id + step to seed randomness for reproducibility in debugging, 
            # but random is fine.
            rand_idx = random.choice(uncovered_indices)
            
            # Get candidate blocks
            candidates = t_subset_k_blocks[rand_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            # Calculate gain for each candidate
            for b_idx in candidates:
                subs = block_t_subsets[b_idx]
                # Gain is number of uncovered subsets in this block
                gain = np.sum(cover_count[subs] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    # Optional: if gain == len(subs), we can break? 
                    # No, another block might have same gain but better properties? 
                    # Gain is the only metric here.
            
            if best_block_idx == -1:
                # Should not happen if uncovered_indices is not empty
                break
                
            # Add block
            current_blocks.append(best_block_idx)
            cover_count[block_t_subsets[best_block_idx]] += 1
            
        # Purge Phase
        # Convert current_blocks to list of indices
        blocks_list = list(current_blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks_list)):
                b_idx = blocks_list[i]
                subs = block_t_subsets[b_idx]
                if np.all(cover_count[subs] > 1):
                    cover_count[subs] -= 1
                    blocks_list.pop(i)
                    changed = True
                    break
        
        if len(blocks_list) < best_len:
            best_len = len(blocks_list)
            # Store the actual blocks (tuples/lists of points)
            best_blocks = [list(all_k_blocks[i]) for i in blocks_list]
            
        run_id += 1
        
    if best_blocks is None:
        # Fallback: return a trivial valid covering (should not happen)
        # Just return all blocks? No, that's huge.
        # The loop should have found something.
        return []
        
    return best_blocks
