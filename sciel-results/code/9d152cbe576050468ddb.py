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
            
    best_block_indices = None
    best_score = float('inf')
    
    restart = 0
    
    while True:
        if time.time() > deadline - 5.0:
            break
            
        random.seed(restart)
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        
        # Greedy Phase with Uncovered List
        uncovered_list = list(range(num_t_subsets))
        
        while len(uncovered_list) > 0:
            # Pick a random uncovered subset
            idx_pos = random.randint(0, len(uncovered_list) - 1)
            u_idx = uncovered_list[idx_pos]
            
            # Get candidates
            candidates = t_subset_to_blocks[u_idx]
            if not candidates:
                break
            
            # Evaluate gain for candidates
            cand_list = candidates
            cand_t_arr = np.vstack([block_t_indices[b] for b in cand_list])
            gains = np.sum(coverage[cand_t_arr] == 0, axis=1)
            
            best_pos = np.argmax(gains)
            best_block_idx = cand_list[best_pos]
            
            # Add block
            block_indices.append(best_block_idx)
            
            # Update coverage
            idxs = block_t_indices[best_block_idx]
            coverage[idxs] += 1
            
            # Update uncovered_list
            # Identify newly covered indices (those that were 0 and are now 1)
            # Since we just incremented, those with value 1 were 0 before.
            newly_covered_set = set(i for i in idxs if coverage[i] == 1)
            if newly_covered_set:
                # Filter uncovered_list
                uncovered_list = [x for x in uncovered_list if x not in newly_covered_set]

        # Reduction Phase (Fast Version)
        current_indices = block_indices[:]
        current_set = set(current_indices)
        
        changed = True
        while changed:
            changed = False
            essential_blocks = set()
            
            # Find t-subsets with coverage == 1
            critical_ts = np.where(coverage == 1)[0]
            
            for t_idx in critical_ts:
                for b_idx in t_subset_to_blocks[t_idx]:
                    if b_idx in current_set:
                        essential_blocks.add(b_idx)
            
            new_indices = [b for b in current_indices if b in essential_blocks]
            
            if len(new_indices) != len(current_indices):
                # Update coverage
                for b in current_indices:
                    if b not in essential_blocks:
                        coverage[block_t_indices[b]] -= 1
                current_indices = new_indices
                current_set = set(new_indices)
                changed = True
            else:
                changed = False
                
        if len(current_indices) < best_score:
            best_score = len(current_indices)
            best_block_indices = current_indices[:]
            
        restart += 1
        
    if best_block_indices is None:
        return []
        
    return [list(all_k_blocks[i]) for i in best_block_indices]
