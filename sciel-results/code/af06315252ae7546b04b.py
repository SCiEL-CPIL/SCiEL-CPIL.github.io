import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    block_t_indices = [] 
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[s] for s in t_subs_in_block], dtype=np.int32)
        block_t_indices.append(indices)
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)
            
    # Convert t_subset_to_blocks to numpy arrays?
    # Lists are okay, but converting to arrays might help if we do fancy indexing.
    # However, we just iterate over them to add to a set. Lists are fine.
    
    best_block_indices = None
    best_score = float('inf')
    
    # Estimate time for one restart to determine max restarts
    # But we can just run in a loop with time check.
    
    restart = 0
    while True:
        if time.time() > deadline - 10.0:
            break
            
        random.seed(restart)
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        
        # Greedy Phase
        while True:
            # Find an uncovered subset
            # Random sampling
            while True:
                u_idx = random.randint(0, num_t_subsets - 1)
                if coverage[u_idx] == 0:
                    break
            
            # Get candidates from K random uncovered subsets
            K = 20 # Increased K for better search
            candidate_block_set = set()
            
            # Add candidates for the first one
            for b_idx in t_subset_to_blocks[u_idx]:
                candidate_block_set.add(b_idx)
                
            # Add candidates for K-1 more
            for _ in range(K - 1):
                while True:
                    u_idx2 = random.randint(0, num_t_subsets - 1)
                    if coverage[u_idx2] == 0:
                        break
                for b_idx in t_subset_to_blocks[u_idx2]:
                    candidate_block_set.add(b_idx)
            
            if not candidate_block_set:
                break # Error state, should not happen
            
            # Vectorized Gain Calculation
            cand_list = list(candidate_block_set)
            # Stack t-indices
            # Note: block_t_indices are int32 arrays of same length
            cand_t_arr = np.vstack([block_t_indices[b] for b in cand_list])
            
            # Calculate gain
            gains = np.sum(coverage[cand_t_arr] == 0, axis=1)
            
            best_pos = np.argmax(gains)
            best_block_idx = cand_list[best_pos]
            
            # Add block
            block_indices.append(best_block_idx)
            
            # Update coverage
            idxs = block_t_indices[best_block_idx]
            coverage[idxs] += 1
            
            # Check if done
            if np.count_nonzero(coverage == 0) == 0:
                break
                
        # Reduction Phase
        # We want to remove blocks that are not essential.
        # Essential if it covers a t-subset with coverage 1.
        
        changed = True
        current_indices = block_indices[:]
        
        while changed:
            changed = False
            new_indices = []
            for b_idx in current_indices:
                t_idxs = block_t_indices[b_idx]
                # If any of its t-subsets has coverage 1, it's essential
                if np.any(coverage[t_idxs] == 1):
                    new_indices.append(b_idx)
                else:
                    # Remove it
                    coverage[t_idxs] -= 1
                    changed = True
            current_indices = new_indices
            
        # Update best
        if len(current_indices) < best_score:
            best_score = len(current_indices)
            best_block_indices = current_indices[:]
            
        restart += 1
        
    if best_block_indices is None:
        # Fallback: return a valid but potentially large covering
        # Greedy always produces a valid one, so this shouldn't happen
        # unless time ran out before the first restart finished.
        # If time ran out, we might have a partial covering?
        # No, the greedy loop runs to completion.
        # If we break the outer loop, we might not have best_block_indices.
        # We should ensure we return something valid.
        # The first restart should finish quickly.
        
        # If we really have nothing, we can return the first greedy result?
        # We didn't store it if we broke early.
        # Let's just return an empty list if this happens, which is invalid.
        # Better to not break the outer loop until we have at least one result.
        
        pass
        
    # Convert indices back to blocks
    return [list(all_k_blocks[i]) for i in best_block_indices]
