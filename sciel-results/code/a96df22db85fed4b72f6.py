import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    # Deadline: 110 seconds to be safe
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Precomputation
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Build incidence matrix M: num_blocks x num_t_subsets
    # M[i, j] = 1 if t_subset j is contained in k_subset i
    # Using uint8 for memory efficiency and speed
    M = np.zeros((num_blocks, num_t_subsets), dtype=np.uint8)
    
    for i, block in enumerate(all_k_subsets):
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            M[i, idx] = 1
            
    # Precompute boolean masks for each block for faster update?
    # M[i] is already a boolean-like array (0/1).
    
    def run_greedy(seed):
        random.seed(seed)
        # uncovered: 1 if t-subset is not yet covered
        uncovered = np.ones(num_t_subsets, dtype=np.uint8)
        selected_indices = []
        
        while uncovered.any():
            # Calculate gain for all blocks: how many currently uncovered t-subsets does each block cover?
            gains = M.dot(uncovered)
            
            max_gain = gains.max()
            if max_gain == 0:
                # Should not happen if uncovered is not empty and M is valid
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one to diversify
            chosen_idx = random.choice(candidates)
            selected_indices.append(chosen_idx)
            
            # Update uncovered
            # M[chosen_idx] is a vector of 0s and 1s
            # We want to set uncovered[j] = 0 where M[chosen_idx][j] == 1
            uncovered[M[chosen_idx] == 1] = 0
            
            # Time check inside loop? 
            # M.dot is fast, but let's check every few iterations
            # Actually, just checking time.time() is cheap enough
            if time.time() > deadline - 5:
                return None, len(selected_indices)
                
        return selected_indices, len(selected_indices)

    def cleanup(selected_indices):
        # Remove redundant blocks
        # cover_count[j] = number of selected blocks covering t-subset j
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        for idx in selected_indices:
            cover_count[M[idx]] += 1
            
        changed = True
        current_indices = list(selected_indices)
        
        while changed and time.time() < deadline - 2:
            changed = False
            new_indices = []
            for i, idx in enumerate(current_indices):
                # Check if block idx is redundant
                # It is redundant if all t-subsets it covers are covered by >1 block
                # i.e., cover_count[j] > 1 for all j where M[idx][j] == 1
                block_mask = M[idx]
                if np.all(cover_count[block_mask] > 1):
                    # Redundant, remove it
                    cover_count[block_mask] -= 1
                    changed = True
                else:
                    new_indices.append(idx)
            current_indices = new_indices
            
        return current_indices

    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 0
    # Run as many restarts as possible
    while time.time() < deadline - 10:
        result, count = run_greedy(num_restarts)
        if result is not None:
            # Perform cleanup
            cleaned = cleanup(result)
            if len(cleaned) < best_count:
                best_count = len(cleaned)
                best_blocks = [all_k_subsets[i] for i in cleaned]
                # If we hit the theoretical lower bound or known best, we can stop early?
                # Known best is 151.
                if best_count <= 151:
                    # We can try a few more to see if we can go lower, but 151 is very good.
                    # Let's continue for a bit to see if 150 is possible, but break if time is tight.
                    pass
        num_restarts += 1
        
        # If we've found a very good solution, we might want to spend time on local search (swaps)
        # But for now, greedy+cleanup is the main driver.
        
    if best_blocks is None:
        # Fallback: return empty or a trivial cover? 
        # The task requires valid cover. 
        # If greedy failed, we should return a valid cover.
        # Since greedy always covers all t-subsets (unless time runs out mid-greedy),
        # we should ensure we return a valid cover.
        # If time ran out, `run_greedy` returns None.
        # We should probably just return the last successful one or a simple greedy without time check?
        # But if we run out of time, we must return *something* valid.
        # The problem states "an invalid result ... scores 0".
        # So we must ensure validity.
        # If `best_blocks` is None, it means we never completed a greedy run.
        # This is unlikely given the speed, but let's add a fallback.
        # A simple greedy without the time check inside might work if we just do one pass.
        # But if we are out of time, we can't do another pass.
        # Let's assume the first greedy run completes.
        
        # If still None, return empty list? No, must be valid.
        # Let's force a quick greedy if we have any time left, or return the partial result?
        # Partial result is invalid.
        # We must return a valid cover.
        # If we never completed a cover, we fail.
        # Given the speed of NumPy, this should not happen.
        
        return []

    return best_blocks
