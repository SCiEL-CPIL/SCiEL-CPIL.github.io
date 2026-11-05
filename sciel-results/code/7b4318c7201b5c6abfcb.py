import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute block to subset indices mapping
    # Shape: (num_blocks, C(k,t))
    k_choose_t = len(list(itertools.combinations(range(k), t)))
    block_subset_indices = np.empty((num_blocks, k_choose_t), dtype=np.int32)
    
    # Precompute subset to block indices mapping (for repair)
    # List of lists, where subset_to_blocks[i] contains indices of blocks covering t_subset i
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, block in enumerate(all_blocks):
        for j, subset in enumerate(itertools.combinations(block, t)):
            idx = t_subset_to_idx[subset]
            block_subset_indices[i, j] = idx
            subset_to_blocks[idx].append(i)
            
    # 2. Greedy Construction (Vectorized)
    uncovered = np.ones(num_t_subsets, dtype=bool)
    selected_blocks = set()
    
    # We use a list for selected blocks for ordering, but a set for O(1) removal
    # For the greedy phase, we just append to a list.
    greedy_selected = []
    
    while uncovered.any():
        if time.time() > deadline - 10: # Reserve time for local search
            break
            
        # Calculate gain: how many uncovered subsets does each block cover?
        # We only care about blocks not yet selected? 
        # For speed, we calculate for all, but ignore selected ones (gain will be 0 if they only cover already covered? No, they cover uncovered ones too, but we shouldn't pick duplicates).
        # Actually, if a block is selected, its subsets are already covered (uncovered=False). So gain for selected blocks is 0.
        gains = uncovered[block_subset_indices].sum(axis=1)
        
        max_gain_idx = np.argmax(gains)
        max_gain = gains[max_gain_idx]
        
        if max_gain == 0:
            break
            
        greedy_selected.append(max_gain_idx)
        selected_blocks.add(max_gain_idx)
        uncovered[block_subset_indices[max_gain_idx]] = False
        
    # 3. Local Search
    # Current solution is greedy_selected
    current_solution = list(greedy_selected)
    best_solution = current_solution.copy()
    best_size = len(current_solution)
    
    # Helper: Calculate coverage counts for a given solution (list of block indices)
    def get_cover_counts(sol):
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in sol:
            counts[block_subset_indices[b_idx]] += 1
        return counts
        
    # Helper: Reduce solution by removing redundant blocks
    def reduce_solution(sol):
        sol = list(sol)
        if not sol:
            return sol
        counts = get_cover_counts(sol)
        changed = True
        while changed:
            if time.time() > deadline - 2:
                break
            changed = False
            new_sol = []
            for b_idx in sol:
                sub_indices = block_subset_indices[b_idx]
                # Check if all subsets in this block are covered by >1 block
                if np.min(counts[sub_indices]) > 1:
                    # Redundant
                    counts[sub_indices] -= 1
                    changed = True
                else:
                    new_sol.append(b_idx)
            sol = new_sol
        return sol
        
    # Initial reduction
    current_solution = reduce_solution(current_solution)
    best_solution = current_solution.copy()
    best_size = len(current_solution)
    
    # Local Search Loop
    # Strategy: Try to remove a block and repair. If repair adds <= 1 block and allows reduction, it's a win.
    # We will try random blocks for removal.
    
    while time.time() < deadline - 5:
        # Try a few random moves
        for _ in range(50):
            if time.time() > deadline - 5:
                break
                
            if not current_solution:
                break
                
            # Pick a random block to remove
            out_idx_pos = random.randint(0, len(current_solution) - 1)
            out_block = current_solution[out_idx_pos]
            
            # Remove it
            new_sol = current_solution.copy()
            new_sol.pop(out_idx_pos)
            
            # Calculate uncovered subsets
            counts = get_cover_counts(new_sol)
            # Find subsets with count 0
            uncovered_indices = np.where(counts == 0)[0]
            
            if len(uncovered_indices) == 0:
                # Removal was successful without repair (it was redundant, but reduce_solution should have caught it? 
                # Maybe it became redundant due to previous moves in this iteration? 
                # Or maybe my reduce_solution missed it? 
                # If counts are all >= 1, it's a valid cover.
                new_sol = reduce_solution(new_sol)
                if len(new_sol) < best_size:
                    best_size = len(new_sol)
                    best_solution = new_sol.copy()
                    current_solution = new_sol.copy()
                    break # Restart reduction loop logic? Or just continue
                elif len(new_sol) < len(current_solution):
                    current_solution = new_sol.copy()
                continue
            
            # Repair: Greedily add blocks to cover uncovered_indices
            # We need to cover the subsets in uncovered_indices
            # We can use a mini-greedy: 
            # 1. Find a block that covers the most of these uncovered subsets.
            # 2. Add it. Update counts.
            # 3. Repeat.
            
            # To speed up, we can limit the number of blocks we try to add.
            # If we need to add > 1 block, it's likely not an improvement (size increases).
            # However, adding 1 block might allow removing 2 others later.
            # So we allow adding up to 2 blocks.
            
            added_blocks = []
            temp_counts = counts.copy()
            temp_uncovered = set(uncovered_indices.tolist())
            
            repair_limit = 2
            for _ in range(repair_limit):
                if not temp_uncovered:
                    break
                
                # Find best block to cover most of temp_uncovered
                # Iterate through all blocks? Too slow.
                # Use subset_to_blocks.
                # Pick a random uncovered subset, get its blocks, check their coverage of other uncovered subsets.
                # Or, iterate through all blocks and score them? 
                # Given 75k blocks, iterating all is ~0.05s in Python if optimized? Maybe.
                # But we can be smarter.
                
                best_block = -1
                best_gain = 0
                
                # Sample a few candidate blocks from the uncovered subsets
                # Pick the first uncovered subset
                sample_sub = next(iter(temp_uncovered))
                candidates = subset_to_blocks[sample_sub]
                
                # Check top 10 candidates?
                # Shuffle candidates to avoid bias?
                random.shuffle(candidates)
                for b_idx in candidates[:20]:
                    if b_idx in new_sol or b_idx in added_blocks:
                        continue
                    # Calculate gain: how many subsets in temp_uncovered does this block cover?
                    sub_indices = block_subset_indices[b_idx]
                    # Convert sub_indices to set for fast intersection?
                    # Or just loop?
                    # len(sub_indices) is 56.
                    gain = 0
                    for s_idx in sub_indices:
                        if s_idx in temp_uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                        
                if best_block == -1 or best_gain == 0:
                    break # Cannot repair? Should not happen if valid cover exists
                
                # Add best_block
                added_blocks.append(best_block)
                # Update temp_counts and temp_uncovered
                sub_indices = block_subset_indices[best_block]
                for s_idx in sub_indices:
                    if s_idx in temp_uncovered:
                        temp_uncovered.remove(s_idx)
                        temp_counts[s_idx] += 1
                    else:
                        temp_counts[s_idx] += 1 # Just in case, though we only track uncovered for repair logic
                
                if len(added_blocks) > 1:
                    # If we had to add more than 1, it's less likely to be an improvement
                    # But we allow up to 2. If we add 2, size increases by 1.
                    # Only accept if it allows reduction.
                    pass

            if len(added_blocks) == 0:
                continue
                
            # Construct new solution
            candidate_sol = new_sol + added_blocks
            # Reduce
            candidate_sol = reduce_solution(candidate_sol)
            
            if len(candidate_sol) < best_size:
                best_size = len(candidate_sol)
                best_solution = candidate_sol.copy()
                current_solution = candidate_sol.copy()
                break # Accept and restart
            elif len(candidate_sol) < len(current_solution):
                current_solution = candidate_sol.copy()
                
        # If no improvement in this batch, perturb
        if time.time() > deadline - 5:
            break
            
        # Perturbation: Remove a random block, repair greedily (allowing more blocks), reduce
        if current_solution:
            out_idx_pos = random.randint(0, len(current_solution) - 1)
            out_block = current_solution[out_idx_pos]
            
            new_sol = current_solution.copy()
            new_sol.pop(out_idx_pos)
            
            counts = get_cover_counts(new_sol)
            uncovered_indices = np.where(counts == 0)[0]
            
            added_blocks = []
            temp_counts = counts.copy()
            temp_uncovered = set(uncovered_indices.tolist())
            
            # Greedy repair with higher limit
            while temp_uncovered:
                sample_sub = next(iter(temp_uncovered))
                candidates = subset_to_blocks[sample_sub]
                random.shuffle(candidates)
                
                best_block = -1
                best_gain = 0
                for b_idx in candidates[:10]:
                    if b_idx in new_sol or b_idx in added_blocks:
                        continue
                    sub_indices = block_subset_indices[b_idx]
                    gain = 0
                    for s_idx in sub_indices:
                        if s_idx in temp_uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                        
                if best_block == -1:
                    break
                    
                added_blocks.append(best_block)
                sub_indices = block_subset_indices[best_block]
                for s_idx in sub_indices:
                    if s_idx in temp_uncovered:
                        temp_uncovered.remove(s_idx)
                    temp_counts[s_idx] += 1
                    
            candidate_sol = new_sol + added_blocks
            candidate_sol = reduce_solution(candidate_sol)
            
            if len(candidate_sol) < len(current_solution):
                current_solution = candidate_sol.copy()
            # We don't update best_solution here unless it's better, to avoid getting stuck in a worse local minimum
            if len(candidate_sol) < best_size:
                best_size = len(candidate_sol)
                best_solution = candidate_sol.copy()
                
    # Return the best solution found
    # Convert indices to lists of points
    result_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return result_blocks
