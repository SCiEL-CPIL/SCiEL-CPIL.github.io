import itertools
import numpy as np
import time
import random
import sys

def construct(v, k, t):
    # Deadline guard
    deadline = time.time() + 115  # 35s margin for safety
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute subsets_of_block: for each block j, list of t-subset indices
    # and blocks_of_subset: for each t-subset i, list of block indices
    subsets_of_block = []
    blocks_of_subset = [[] for _ in range(n_t)]
    
    for j, B in enumerate(k_subsets):
        sub_indices = []
        for S in itertools.combinations(B, t):
            i = t_index[S]
            sub_indices.append(i)
            blocks_of_subset[i].append(j)
        subsets_of_block.append(np.array(sub_indices, dtype=np.int32))
        
    # Convert blocks_of_subset to arrays for faster iteration? 
    # Or keep as lists if we slice them? Arrays are better for vectorized ops, but here we iterate.
    # Let's keep as lists for now, as we just need to update gains.

    best_solution = None
    best_size = float('inf')
    
    # Multi-start greedy
    # We will run several iterations.
    # To add randomness, we can shuffle the initial gains or the order of tie-breaking.
    # Or we can just run it a few times, as the greedy is deterministic unless we add noise.
    
    # Let's add a small random noise to the initial gains to break ties and explore different paths.
    # Or randomize the order of blocks considered?
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        uncovered_mask = np.ones(n_t, dtype=bool)
        gains = np.array([len(subs) for subs in subsets_of_block], dtype=np.float64)
        
        # Add small random noise to break symmetry/ties
        if restart > 0:
            gains += np.random.uniform(0, 0.1, n_k)
            
        solution_blocks = []
        
        while True:
            if time.time() > deadline:
                break
                
            # Find best block
            if np.max(gains) <= 0:
                break
                
            best_idx = np.argmax(gains)
            # If gain is 0, we are stuck (shouldn't happen if uncovered exists)
            if gains[best_idx] == 0:
                 # Fallback: pick any block that covers something? 
                 # If max gain is 0 but uncovered exists, it means we have a bug or logical error.
                 # Actually, if max gain is 0, it means no block covers any uncovered subset? 
                 # That's impossible if uncovered is not empty.
                 break

            solution_blocks.append(best_idx)
            
            # Update gains
            subs_b = subsets_of_block[best_idx]
            
            # Find which subsets in best_idx are actually uncovered
            # We can check uncovered_mask[subs_b]
            newly_covered_mask = uncovered_mask[subs_b]
            
            if not np.any(newly_covered_mask):
                # This block covers nothing new? Should not happen if gain > 0.
                # Gain was calculated based on uncovered.
                pass
                
            # Update gains for all blocks sharing subsets with best_idx
            # We iterate over the newly covered subsets
            for s_idx in subs_b[newly_covered_mask]:
                # Get all blocks containing s_idx
                for b_prime in blocks_of_subset[s_idx]:
                    gains[b_prime] -= 1.0
                    
            # Update uncovered_mask
            uncovered_mask[subs_b] = False
            
            # Check if all covered
            if not np.any(uncovered_mask):
                break
                
        # Solution found (or best effort)
        if len(solution_blocks) < best_size:
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()
            
            # If we reached a very good size, we might stop early?
            # 151 is the target. If we hit 151, keep searching for 150.
            
    # Post-processing: Deletion Pass on best_solution
    if best_solution is not None:
        # Compute cover counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1
            
        # Try to remove blocks
        changed = True
        while changed:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                # Check if any subset in b_idx has count 1
                # If count > 1 for all, we can remove b_idx
                # If any count == 1, we MUST keep b_idx
                can_remove = True
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove b_idx
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)
                    
            best_solution = new_solution
            
    # Construct the output list of blocks
    if best_solution is None:
        # Emergency fallback: simple greedy without optimization
        # This should not be reached if the loop ran
        uncovered = set(range(n_t))
        res_blocks = []
        while uncovered:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = 0
                for s in subsets_of_block[j]:
                    if s in uncovered:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_b = j
            if best_b == -1 or max_cov == 0:
                break
            res_blocks.append(best_b)
            for s in subsets_of_block[best_b]:
                if s in uncovered:
                    uncovered.remove(s)
        best_solution = res_blocks

    # Return list of lists of ints
    return [list(k_subsets[idx]) for idx in best_solution]
