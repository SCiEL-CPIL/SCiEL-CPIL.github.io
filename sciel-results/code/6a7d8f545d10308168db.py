import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute coverage for each block: list of t-subset indices
    block_cover_indices = []
    for block in k_blocks:
        subs = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[sub] for sub in subs], dtype=np.int32)
        block_cover_indices.append(indices)
        
    # Precompute inverse: for each t-subset, which blocks cover it
    # This is useful for repair, but for SA swaps, we just need block_cover_indices
    # We will use a boolean array for membership in current solution to check "in solution" quickly
    
    def greedy_construction():
        uncovered = np.ones(n_t_subsets, dtype=bool)
        selected = []
        # Initial gains: C(k, t) for all
        gains = np.full(n_k_blocks, len(block_cover_indices[0]), dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline - 30: # Reserve time for SA
                break
            best_block = np.argmax(gains)
            if gains[best_block] == 0:
                break
            selected.append(best_block)
            
            # Update gains
            for s_idx in block_cover_indices[best_block]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    # Decrement gain for all blocks covering s_idx
                    # We need the list of blocks covering s_idx
                    # We didn't precompute tsub_to_blocks_np in this snippet, let's add it or compute on fly?
                    # Computing on fly is slow. Let's precompute tsub_to_blocks_np outside.
        return selected

    # Precompute tsub_to_blocks_np for greedy update speed
    tsub_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, indices in enumerate(block_cover_indices):
        for s_idx in indices:
            tsub_to_blocks[s_idx].append(b_idx)
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]

    def greedy_construction_v2():
        uncovered = np.ones(n_t_subsets, dtype=bool)
        selected = []
        gains = np.full(n_k_blocks, len(block_cover_indices[0]), dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline - 30:
                break
            best_block = np.argmax(gains)
            if gains[best_block] == 0:
                break
            selected.append(best_block)
            
            for s_idx in block_cover_indices[best_block]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    gains[tsub_to_blocks_np[s_idx]] -= 1
        return selected

    # Initial Solution via Greedy
    initial_indices = greedy_construction_v2()
    
    # Verify initial solution
    if not initial_indices:
        # Fallback if greedy failed (shouldn't happen)
        return [list(k_blocks[i]) for i in range(n_k_blocks)] # Worst case

    # Setup for Simulated Annealing
    # State: set of selected block indices
    # We use a boolean array `in_solution` for O(1) check
    in_solution = np.zeros(n_k_blocks, dtype=bool)
    for idx in initial_indices:
        in_solution[idx] = True
        
    # Coverage counts
    coverage = np.zeros(n_t_subsets, dtype=np.int32)
    for idx in initial_indices:
        coverage[block_cover_indices[idx]] += 1
        
    # Current Cost: len + penalty * uncovered
    # We want to minimize len, but uncovered must be 0.
    # Penalty should be high enough that we never accept a state with uncovered > 0 if we can avoid it,
    # but SA needs to be able to go "up" in cost to escape.
    # Let's define Cost = len(selected) + 1000 * sum(coverage == 0)
    
    def get_cost(selected_indices, coverage_arr):
        uncovered_count = np.sum(coverage_arr == 0)
        return len(selected_indices) + 1000 * uncovered_count

    current_indices = list(initial_indices)
    current_cost = get_cost(current_indices, coverage)
    best_indices = list(current_indices)
    best_cost = current_cost
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing Parameters
    T = 10.0 # Initial Temperature
    T_min = 0.1
    cooling_rate = 0.995 # Slow cooling for long run
    max_iter = 5000000 # Cap iterations, time is the real limit
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate Neighbor
        # Strategy: 
        # 1. Try Swap (remove one, add one)
        # 2. Try 2-Swap (remove two, add two) - more powerful but slower
        
        move_type = rng.random()
        if move_type < 0.7:
            # Swap
            # Pick a block to remove
            if not current_indices:
                continue
            rem_idx_pos = rng.integers(0, len(current_indices))
            rem_block = current_indices[rem_idx_pos]
            
            # Pick a block to add (not in solution)
            # We can't just pick random from 0..n_k_blocks and check in_solution every time if n_k_blocks is large?
            # n_k_blocks = 50k. in_solution is a bool array. Check is O(1).
            # But finding a random "not in solution" index efficiently?
            # We can just pick random and check. Probability of hitting an in-solution block is ~150/50000 = 0.3%. Very low.
            add_block = rng.integers(0, n_k_blocks)
            if in_solution[add_block]:
                # Retry a few times or just skip
                # For simplicity, just skip this move
                continue
                
            # Calculate Delta
            # Remove rem_block
            rem_subs = block_cover_indices[rem_block]
            add_subs = block_cover_indices[add_block]
            
            # Check if rem_block is essential (covers any subset with count 1)
            # If yes, uncovered count will increase.
            # We need to calculate new coverage counts locally?
            # No, we can just update coverage array, check cost, and revert if rejected.
            
            # Apply Move
            coverage[rem_subs] -= 1
            coverage[add_subs] += 1
            in_solution[rem_block] = False
            in_solution[add_block] = True
            current_indices[rem_idx_pos] = add_block # Keep list size same for swap
            
            new_cost = get_cost(current_indices, coverage)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_indices = list(current_indices)
            else:
                # Revert
                coverage[add_subs] -= 1
                coverage[rem_subs] += 1
                in_solution[add_block] = False
                in_solution[rem_block] = True
                current_indices[rem_idx_pos] = rem_block
                
        elif move_type < 0.9:
            # 2-Swap: Remove 2, Add 2
            if len(current_indices) < 2:
                continue
            # Pick 2 distinct positions to remove
            rem_pos1, rem_pos2 = rng.choice(len(current_indices), size=2, replace=False)
            rem_block1 = current_indices[rem_pos1]
            rem_block2 = current_indices[rem_pos2]
            
            add_block1 = rng.integers(0, n_k_blocks)
            add_block2 = rng.integers(0, n_k_blocks)
            
            if in_solution[add_block1] or in_solution[add_block2] or add_block1 == add_block2:
                continue
            if add_block1 == rem_block1 or add_block1 == rem_block2 or add_block2 == rem_block1 or add_block2 == rem_block2:
                continue
                
            # Apply
            coverage[block_cover_indices[rem_block1]] -= 1
            coverage[block_cover_indices[rem_block2]] -= 1
            coverage[block_cover_indices[add_block1]] += 1
            coverage[block_cover_indices[add_block2]] += 1
            
            in_solution[rem_block1] = False
            in_solution[rem_block2] = False
            in_solution[add_block1] = True
            in_solution[add_block2] = True
            
            current_indices[rem_pos1] = add_block1
            current_indices[rem_pos2] = add_block2
            
            new_cost = get_cost(current_indices, coverage)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_indices = list(current_indices)
            else:
                # Revert
                coverage[block_cover_indices[add_block1]] -= 1
                coverage[block_cover_indices[add_block2]] -= 1
                coverage[block_cover_indices[rem_block1]] += 1
                coverage[block_cover_indices[rem_block2]] += 1
                
                in_solution[add_block1] = False
                in_solution[add_block2] = False
                in_solution[rem_block1] = True
                in_solution[rem_block2] = True
                
                current_indices[rem_pos1] = rem_block1
                current_indices[rem_pos2] = rem_block2
                
        else:
            # Add Move (increase size, useful if we are stuck with high uncovered)
            # Actually, if uncovered > 0, we should try to add a block that covers many uncovered.
            if np.any(coverage == 0):
                # Find block covering most uncovered
                # This is expensive. 
                # Alternative: Random add.
                add_block = rng.integers(0, n_k_blocks)
                if in_solution[add_block]:
                    continue
                
                coverage[block_cover_indices[add_block]] += 1
                in_solution[add_block] = True
                current_indices.append(add_block)
                
                new_cost = get_cost(current_indices, coverage)
                delta = new_cost - current_cost
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_indices = list(current_indices)
                else:
                    coverage[block_cover_indices[add_block]] -= 1
                    in_solution[add_block] = False
                    current_indices.pop()
            
            # Remove Move (decrease size, if redundant)
            elif current_indices:
                # Pick random block to remove
                rem_idx_pos = rng.integers(0, len(current_indices))
                rem_block = current_indices[rem_idx_pos]
                
                # Check if redundant
                if np.min(coverage[block_cover_indices[rem_block]]) > 1:
                    coverage[block_cover_indices[rem_block]] -= 1
                    in_solution[rem_block] = False
                    current_indices.pop(rem_idx_pos)
                    current_cost -= 1 # Cost decreases by 1, uncovered unchanged
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_indices = list(current_indices)
                else:
                    # Not redundant, but we can still try to remove it and see if SA accepts the penalty?
                    # No, that increases uncovered. Only accept if T is high.
                    coverage[block_cover_indices[rem_block]] -= 1
                    in_solution[rem_block] = False
                    current_indices.pop(rem_idx_pos)
                    
                    new_cost = get_cost(current_indices, coverage)
                    delta = new_cost - current_cost
                    
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_indices = list(current_indices)
                    else:
                        coverage[block_cover_indices[rem_block]] += 1
                        in_solution[rem_block] = True
                        current_indices.insert(rem_idx_pos, rem_block)

        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final Polish: Greedy Removal of Redundant Blocks
    # The SA might leave redundant blocks.
    final_indices = list(best_indices)
    # Recalculate coverage for final_indices
    coverage_final = np.zeros(n_t_subsets, dtype=np.int32)
    for idx in final_indices:
        coverage_final[block_cover_indices[idx]] += 1
        
    # Remove redundant
    new_final = []
    for idx in final_indices:
        subs = block_cover_indices[idx]
        if np.min(coverage_final[subs]) > 1:
            coverage_final[subs] -= 1
        else:
            new_final.append(idx)
    
    # Verify
    if np.any(coverage_final == 0):
        # If SA broke validity (shouldn't happen if we track best_cost correctly with penalty),
        # fall back to initial greedy or re-run greedy on the remaining uncovered
        # For safety, if invalid, use the initial greedy result
        # But best_cost tracks the penalty, so best_indices should be valid if best_cost < 1000
        # If best_cost >= 1000, it's invalid.
        if best_cost >= 1000:
             # Use initial greedy
             final_indices = initial_indices
        else:
             # Try to repair
             # Find uncovered
             uncovered_mask = (coverage_final == 0)
             # Greedily add blocks to cover them
             for _ in range(100):
                 if not np.any(uncovered_mask):
                     break
                 # Find block covering most uncovered
                 best_b = -1
                 best_gain = 0
                 # Sample some blocks
                 sample = rng.choice(n_k_blocks, size=100, replace=False)
                 for b in sample:
                     subs = block_cover_indices[b]
                     gain = np.sum(uncovered_mask[subs])
                     if gain > best_gain:
                         best_gain = gain
                         best_b = b
                 if best_b != -1:
                     final_indices.append(best_b)
                     coverage_final[block_cover_indices[best_b]] += 1
                     uncovered_mask = (coverage_final == 0)
    
    return [list(k_blocks[i]) for i in final_indices]
