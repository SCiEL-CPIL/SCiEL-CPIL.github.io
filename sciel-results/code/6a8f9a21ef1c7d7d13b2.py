import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145
    
    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    masks = [0] * num_k_subsets
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    all_ones = (1 << num_t_subsets) - 1
    
    best_solution = None
    best_size = float('inf')
    
    # Run multiple greedy restarts if time permits
    # First run is deterministic
    for restart in range(20): # Max 20 restarts
        if time.time() > deadline - 10: # Leave 10s for final processing
            break
            
        random.seed(restart) # Deterministic restarts for reproducibility
        
        uncovered = all_ones
        selected_blocks = []
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline - 5:
                break
                
            best_gain = -1
            best_candidates = []
            
            for i in range(num_k_subsets):
                gain = (uncovered & masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
                    
            if not best_candidates:
                break
                
            # Pick a candidate
            # If restart == 0, pick first. Else pick random.
            if restart == 0:
                idx = best_candidates[0]
            else:
                idx = random.choice(best_candidates)
                
            selected_blocks.append(k_subsets[idx])
            selected_indices.append(idx)
            uncovered &= ~masks[idx]
            
        # Local Search: Deletion
        if not selected_blocks:
            continue
            
        # Calculate cover counts
        cover_count = [0] * num_t_subsets
        for block in selected_blocks:
            for t_sub in itertools.combinations(block, t):
                idx = t_subset_to_idx[t_sub]
                cover_count[idx] += 1
                
        changed = True
        while changed:
            if time.time() > deadline - 2:
                break
            changed = False
            new_selected_blocks = []
            for i, block in enumerate(selected_blocks):
                essential = False
                for t_sub in itertools.combinations(block, t):
                    idx = t_subset_to_idx[t_sub]
                    if cover_count[idx] == 1:
                        essential = True
                        break
                
                if essential:
                    new_selected_blocks.append(block)
                else:
                    changed = True
                    for t_sub in itertools.combinations(block, t):
                        idx = t_subset_to_idx[t_sub]
                        cover_count[idx] -= 1
                        
            if changed:
                selected_blocks = new_selected_blocks
                
        # Update best
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = selected_blocks
            # If we reached the theoretical lower bound or a very good number, we might stop early
            # Lower bound is ~208. Known best 335.
            # If we get < 300, that's great.
            
    if best_solution is None:
        # Fallback: return any valid solution or empty? 
        # The problem guarantees a solution exists. 
        # If we timed out during the first greedy, we might have a partial list.
        # But we should ensure validity.
        # If best_solution is None, it means we didn't finish even one greedy pass?
        # Unlikely. 
        return []
        
    return best_solution
