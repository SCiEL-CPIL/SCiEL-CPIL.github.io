import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # For each block, precompute the indices of t-subsets it covers
    block_t_indices = []
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[s] for s in t_subs_in_block], dtype=np.int32)
        block_t_indices.append(indices)
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)
    
    # Precompute candidate lists as numpy arrays for faster access
    t_subset_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in t_subset_to_blocks]
    
    best_block_indices = None
    best_score = float('inf')
    
    restart = 0
    while True:
        if time.time() > deadline - 20.0:
            break
            
        random.seed(restart * 7919 + 13)
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        uncovered_list = list(range(num_t_subsets))
        
        # Greedy Phase
        while uncovered_list:
            if time.time() > deadline - 5.0:
                break
            
            # Adaptive K
            frac_uncovered = len(uncovered_list) / num_t_subsets
            if frac_uncovered > 0.5:
                K = min(50, max(10, int(50 * frac_uncovered)))
            elif frac_uncovered > 0.1:
                K = 20
            else:
                K = 10
            
            # Pick K random uncovered subsets
            sampled = random.sample(uncovered_list, min(K, len(uncovered_list)))
            
            # Collect candidate blocks
            candidate_set = set()
            for u_idx in sampled:
                for b_idx in t_subset_to_blocks_arr[u_idx]:
                    candidate_set.add(b_idx)
            
            if not candidate_set:
                break
            
            cand_list = list(candidate_set)
            
            # Vectorized gain calculation
            # Stack all t-indices for candidates
            # This can be memory heavy if cand_list is large, but K is small so cand_list is limited
            # Max cand_list size is sum of blocks covering K subsets.
            # For C(19,8,5), each 5-subset is in C(14,3)=364 blocks.
            # If K=10, max candidates ~ 3640. 
            # 3640 * 126 (t-subsets per block) ~ 450k ints. OK.
            
            cand_t_arr = np.vstack([block_t_indices[b] for b in cand_list])
            
            # Gain = number of currently uncovered t-subsets covered by this block
            gains = np.sum(coverage[cand_t_arr] == 0, axis=1)
            
            best_pos = np.argmax(gains)
            best_block_idx = cand_list[best_pos]
            
            # Add block
            block_indices.append(best_block_idx)
            
            # Update coverage
            idxs = block_t_indices[best_block_idx]
            was_uncovered = coverage[idxs] == 0
            coverage[idxs] += 1
            
            # Update uncovered list
            newly_covered = idxs[was_uncovered]
            if len(newly_covered) > 0:
                newly_covered_set = set(newly_covered.tolist())
                uncovered_list = [x for x in uncovered_list if x not in newly_covered_set]
            
            if not uncovered_list:
                break
        
        # If we broke out of greedy due to time, this covering might be invalid
        if uncovered_list:
            # Try to complete quickly with simple greedy
            while uncovered_list and time.time() < deadline - 2.0:
                u_idx = random.choice(uncovered_list)
                candidates = t_subset_to_blocks_arr[u_idx]
                # Pick candidate with max gain
                best_b = -1
                best_gain = -1
                for b_idx in candidates:
                    gain = np.count_nonzero(coverage[block_t_indices[b_idx]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1: break
                block_indices.append(best_b)
                idxs = block_t_indices[best_b]
                was_uncovered = coverage[idxs] == 0
                coverage[idxs] += 1
                newly_covered = idxs[was_uncovered]
                if len(newly_covered) > 0:
                    newly_covered_set = set(newly_covered.tolist())
                    uncovered_list = [x for x in uncovered_list if x not in newly_covered_set]
        
        if uncovered_list:
            continue # Invalid covering, restart
        
        # Reduction Phase: remove non-essential blocks
        # Use a set for O(1) lookup? No, we need to iterate.
        # Randomize order for reduction to escape symmetries
        current_indices = block_indices[:]
        random.shuffle(current_indices)
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            # Shuffle again each pass? Or just iterate.
            for b_idx in current_indices:
                t_idxs = block_t_indices[b_idx]
                # Essential if it covers at least one t-subset with coverage exactly 1
                if np.any(coverage[t_idxs] == 1):
                    new_indices.append(b_idx)
                else:
                    # Remove it
                    coverage[t_idxs] -= 1
                    changed = True
            current_indices = new_indices
            
            # Early exit if time is tight
            if time.time() > deadline - 10.0:
                break
        
        # Local Search Phase
        # Try to swap blocks to reduce count
        # Strategy: 
        # 1. Identify a block B to remove.
        # 2. Find subsets L that become uncovered.
        # 3. Find a replacement B' that covers L.
        # 4. If B' is found, perform swap.
        # 5. Check if any other block B'' is now redundant.
        # 6. If yes, remove B''. Net gain: -1 block.
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Randomize order of blocks to try
            block_order = current_indices[:]
            random.shuffle(block_order)
            
            for i, b_idx in enumerate(block_order):
                if i >= len(current_indices): continue # List might have changed? No, we are iterating over a copy.
                # But if we modify current_indices, indices in block_order might be stale.
                # Better to iterate over indices of current_indices directly.
                pass
            
            # Let's restart the loop structure to be safe with index changes
            for i in range(len(current_indices)):
                if time.time() > deadline - 5.0: break
                b_idx = current_indices[i]
                t_idxs = block_t_indices[b_idx]
                
                # Find subsets that would become uncovered if we remove this block
                at_risk_mask = coverage[t_idxs] == 1
                at_risk = t_idxs[at_risk_mask]
                
                if len(at_risk) == 0:
                    # Can remove entirely
                    coverage[t_idxs] -= 1
                    current_indices.pop(i)
                    improved = True
                    break # Restart reduction/local search loop
                
                # If at_risk is not empty, try to find a replacement
                # Only try if at_risk is small enough to search
                if len(at_risk) > 5:
                    continue
                
                # Find candidate replacements that cover all at_risk
                # Intersection of block lists for each at_risk subset
                candidates = set(t_subset_to_blocks_arr[at_risk[0]].tolist())
                for r in at_risk[1:]:
                    candidates &= set(t_subset_to_blocks_arr[r].tolist())
                
                candidates.discard(b_idx)
                
                if not candidates:
                    continue
                
                # Evaluate candidates
                # We want a candidate B' such that after swapping B -> B', 
                # we can remove another block B''.
                # Or simply, B' covers more "redundant" subsets, making them easier to remove?
                # Actually, the condition for removing B'' is that B'' has no unique subsets.
                
                best_repl = -1
                best_score_repl = -1 # We want to maximize potential for removal
                
                for rep_b in candidates:
                    rep_t_idxs = block_t_indices[rep_b]
                    
                    # Simulate swap:
                    # 1. Remove B: coverage[t_idxs] -= 1
                    # 2. Add B': coverage[rep_t_idxs] += 1
                    
                    # We need to check if any block B'' (other than B and B') becomes redundant.
                    # A block B'' is redundant if for all s in B'', coverage_new[s] > 1.
                    
                    # This check is expensive if done for all B''.
                    # Optimization: 
                    # The only blocks that can change their redundancy status are:
                    # 1. Blocks that share subsets with B (lost coverage) -> might become MORE redundant? No, losing coverage makes them LESS likely to be redundant (more unique subsets).
                    # 2. Blocks that share subsets with B' (gained coverage) -> might become MORE redundant.
                    
                    # So we only need to check blocks that intersect B'.
                    # Let's collect candidates for B''.
                    # B'' must be in current_indices.
                    # And B'' must share a subset with B'.
                    
                    # For simplicity and speed, we can just check if the swap itself is valid (covers at_risk) 
                    # and if it doesn't increase the count (it doesn't, it's a swap).
                    # But we want to REDUCE the count.
                    # So we need to find a B'' that becomes redundant.
                    
                    # Let's approximate: 
                    # If B' covers many subsets that are currently covered by exactly 1 other block, 
                    # it might not help. 
                    # If B' covers subsets that are covered by many blocks, it might help make them redundant? No.
                    
                    # Actually, the most common improvement is:
                    # Swap B for B'.
                    # Then, check if any block B'' in the set is now redundant.
                    # We can just do a quick check on a few random blocks or blocks intersecting B'.
                    
                    # Let's try a simpler heuristic:
                    # Just take the first valid replacement. 
                    # The reduction phase will catch any new redundancies in the next iteration.
                    # So, if we just swap, and then run reduction, we might reduce.
                    
                    # However, doing a full reduction pass is expensive.
                    # Let's just do the swap and set improved=True, and let the outer loop (which includes reduction) handle the rest.
                    # Wait, the current loop structure is:
                    # while improved:
                    #   for i...
                    #     if remove: improved=True, break
                    #     if swap: improved=True, break?
                    
                    # If we swap, we should break and restart the reduction/loop to see if removal is possible.
                    
                    best_repl = rep_b
                    break # Take first valid replacement for speed
                
                if best_repl != -1:
                    # Perform swap
                    coverage[t_idxs] -= 1
                    new_t_idxs = block_t_indices[best_repl]
                    coverage[new_t_idxs] += 1
                    current_indices[i] = best_repl
                    improved = True
                    break # Restart to check for removals
        
        # Final check for redundancy after local search
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            new_indices = []
            for b_idx in current_indices:
                t_idxs = block_t_indices[b_idx]
                if np.any(coverage[t_idxs] == 1):
                    new_indices.append(b_idx)
                else:
                    coverage[t_idxs] -= 1
                    changed = True
            current_indices = new_indices
        
        # Update best
        if len(current_indices) < best_score:
            best_score = len(current_indices)
            best_block_indices = current_indices[:]
            
        restart += 1
        
        # If we've done several restarts and time is running out, stop
        if restart > 100:
            break
    
    if best_block_indices is None:
        # Fallback: run one more greedy to ensure we have a valid covering
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        uncovered_list = list(range(num_t_subsets))
        
        while uncovered_list:
            u_idx = random.choice(uncovered_list)
            candidates = t_subset_to_blocks_arr[u_idx]
            best_block_idx = -1
            best_gain = -1
            for b_idx in candidates:
                gain = np.count_nonzero(coverage[block_t_indices[b_idx]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            block_indices.append(best_block_idx)
            idxs = block_t_indices[best_block_idx]
            was_uncovered = coverage[idxs] == 0
            coverage[idxs] += 1
            newly_covered = idxs[was_uncovered]
            if len(newly_covered) > 0:
                newly_covered_set = set(newly_covered.tolist())
                uncovered_list = [x for x in uncovered_list if x not in newly_covered_set]
        
        best_block_indices = block_indices
    
    # Convert indices back to blocks
    return [list(all_k_blocks[i]) for i in best_block_indices]
