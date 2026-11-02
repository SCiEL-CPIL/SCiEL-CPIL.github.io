import itertools
import random
import time
import math
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, store the list of t-subset IDs it covers
    # Using numpy arrays for faster vectorized operations later
    block_t_ids = []
    for block in all_k_subsets:
        ids = np.array([t_subset_to_id[s] for s in itertools.combinations(block, t)], dtype=np.int32)
        block_t_ids.append(ids)
        
    # For each t-subset ID, store the list of k-subset indices that cover it
    # This is needed for greedy initialization and repair
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, ids in enumerate(block_t_ids):
        for tid in ids:
            t_to_blocks[tid].append(b_idx)

    # 1. Greedy Construction for Initial Solution
    def greedy_cover():
        is_covered = np.zeros(num_t_subsets, dtype=bool)
        selected_blocks = set()
        
        # To speed up greedy, we can maintain a list of uncovered indices
        # But updating it efficiently is tricky. 
        # Since num_t_subsets is ~11k, scanning is O(11k). 
        # Greedy steps are usually < 1000. 11k * 1000 = 11M ops. Fast enough.
        
        while True:
            # Find uncovered t-subset
            # Using numpy to find first uncovered is fast
            uncovered_mask = ~is_covered
            if not np.any(uncovered_mask):
                break
                
            # Pick a random uncovered subset
            # Get indices of uncovered
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            target_tid = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
            
            # Find block covering target_tid that covers most other uncovered subsets
            candidates = t_to_blocks[target_tid]
            best_block = -1
            max_new_cover = -1
            
            for b_idx in candidates:
                if b_idx in selected_blocks:
                    continue
                # Count how many uncovered t-subsets this block covers
                # Vectorized count
                block_ids = block_t_ids[b_idx]
                # Check which of these are currently uncovered
                # is_covered is a boolean array. ~is_covered is uncovered.
                # We need to check if the specific ids in block_ids are in the uncovered set.
                # Since block_ids are indices, we can index is_covered directly.
                new_cover = np.sum(~is_covered[block_ids])
                
                if new_cover > max_new_cover:
                    max_new_cover = new_cover
                    best_block = b_idx
            
            if best_block != -1:
                selected_blocks.add(best_block)
                # Update is_covered
                is_covered[block_t_ids[best_block]] = True
            else:
                break
                
        return selected_blocks

    # Get initial solution
    initial_solution = greedy_cover()
    
    if not initial_solution:
        return [list(all_k_subsets[0])]

    # 2. Simulated Annealing Optimization with NumPy
    active_blocks = set(initial_solution)
    is_active = np.zeros(num_k_subsets, dtype=bool)
    for b_idx in active_blocks:
        is_active[b_idx] = True
        
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in active_blocks:
        cover_counts[block_t_ids[b_idx]] += 1
        
    uncovered_count = np.sum(cover_counts == 0)
    
    # Parameters for SA
    T = 50.0
    T_min = 0.5
    cooling_rate = 0.995
    penalty = 1000
    
    best_solution = set(active_blocks)
    best_uncovered = uncovered_count
    best_size = len(active_blocks)
    
    # Precompute a list of all block indices for random selection
    # random.randint is fast, but checking is_active is O(1) with numpy bool array
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Try Delete
            if not active_blocks:
                continue
            
            # Efficiently pick a random active block
            # Instead of list(active_blocks), we can pick a random index and check is_active
            # But rejection sampling might be slow if active_blocks is small.
            # Since active_blocks is usually large, rejection is fine.
            # Or we can maintain a list of active blocks and update it.
            # Let's stick to rejection sampling for simplicity, it's O(1) average if density is reasonable.
            # However, if we delete a lot, density drops.
            # Let's use a list for active blocks to allow O(1) random choice.
            # We need to keep this list in sync.
            
            # Actually, maintaining a list is safer for O(1) choice.
            # But updating the list on add/remove is O(N) if we use list.remove.
            # Let's use a set for membership and a list for iteration/choice?
            # No, let's just use random.randint and is_active check. 
            # If the fraction of active blocks is low, we waste time.
            # Let's assume the solution size is a small fraction of all blocks.
            # C(19,8) = 75582. Solution ~ 300. Fraction ~ 0.4%.
            # Rejection sampling would take ~250 tries on average. That's too slow.
            
            # We MUST maintain a list of active block indices for efficient random choice.
            active_list = list(active_blocks)
            b_idx = active_list[random.randint(0, len(active_list)-1)]
            
            # Calculate delta
            b_ids = block_t_ids[b_idx]
            # delta_uncovered: number of tids where count was 1 and becomes 0
            delta_uncovered = np.sum(cover_counts[b_ids] == 1)
            
            delta_E = -1 + penalty * delta_uncovered
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                active_blocks.remove(b_idx)
                is_active[b_idx] = False
                cover_counts[b_ids] -= 1
                uncovered_count += delta_uncovered
                
                if uncovered_count == 0 and len(active_blocks) < best_size:
                    best_size = len(active_blocks)
                    best_solution = set(active_blocks)
                    best_uncovered = 0
                    
        elif move_type < 0.8:
            # Try Insert
            # Pick random inactive block
            # Rejection sampling: pick random index, check is_active
            # Fraction inactive is high (~99.6%), so this is very fast.
            while True:
                b_idx = random.randint(0, num_k_subsets - 1)
                if not is_active[b_idx]:
                    break
            
            b_ids = block_t_ids[b_idx]
            # delta_uncovered: number of tids where count was 0 and becomes >0
            # Actually, we just need to know how many new covers we get.
            # new_cover = sum(cover_counts[b_ids] == 0)
            new_cover = np.sum(cover_counts[b_ids] == 0)
            
            delta_E = 1 - penalty * new_cover
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                active_blocks.add(b_idx)
                is_active[b_idx] = True
                cover_counts[b_ids] += 1
                uncovered_count -= new_cover
                
                if uncovered_count == 0 and len(active_blocks) < best_size:
                    best_size = len(active_blocks)
                    best_solution = set(active_blocks)
                    best_uncovered = 0
                    
        else:
            # Try Swap
            if not active_blocks:
                continue
            
            active_list = list(active_blocks)
            b_in = active_list[random.randint(0, len(active_list)-1)]
            
            # Pick random inactive block for b_out
            while True:
                b_out = random.randint(0, num_k_subsets - 1)
                if not is_active[b_out]:
                    break
            
            if b_in == b_out:
                continue
            
            b_ids_in = block_t_ids[b_in]
            b_ids_out = block_t_ids[b_out]
            
            # Calculate delta_uncovered for swap
            # We need to simulate the change.
            # For each tid in union(b_ids_in, b_ids_out):
            #   old_count = cover_counts[tid]
            #   new_count = old_count - (1 if tid in b_ids_in else 0) + (1 if tid in b_ids_out else 0)
            #   if old_count > 0 and new_count == 0: delta += 1
            #   if old_count == 0 and new_count > 0: delta -= 1
            
            # To do this vectorized:
            # We can create a temporary array of counts for the affected tids.
            # But the tids are not contiguous.
            # We can just iterate over the smaller set or use numpy fancy indexing.
            
            # Let's combine the IDs
            # Since the sets are small (56 each), we can just use a set logic in Python for the delta calculation?
            # 56+56 = 112 elements. Python loop over 112 elements is very fast.
            # Vectorized numpy might have overhead for such small arrays.
            
            # Let's try a Python loop for the swap delta, it might be faster than numpy setup for small N.
            delta_uncovered = 0
            # To avoid double counting, we can iterate over the union
            # Or just iterate over b_ids_in and b_ids_out separately and handle overlaps?
            # Easier: Use a set for the affected tids.
            affected_tids = set(b_ids_in.tolist())
            affected_tids.update(b_ids_out.tolist())
            
            for tid in affected_tids:
                c_old = cover_counts[tid]
                in_in = 1 if tid in b_ids_in else 0
                in_out = 1 if tid in b_ids_out else 0
                c_new = c_old - in_in + in_out
                
                if c_old > 0 and c_new == 0:
                    delta_uncovered += 1
                elif c_old == 0 and c_new > 0:
                    delta_uncovered -= 1
            
            delta_E = -penalty * delta_uncovered
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                active_blocks.remove(b_in)
                is_active[b_in] = False
                cover_counts[b_ids_in] -= 1
                
                active_blocks.add(b_out)
                is_active[b_out] = True
                cover_counts[b_ids_out] += 1
                
                uncovered_count += delta_uncovered
                
                if uncovered_count == 0 and len(active_blocks) < best_size:
                    best_size = len(active_blocks)
                    best_solution = set(active_blocks)
                    best_uncovered = 0

        T *= cooling_rate

    # 3. Repair Phase
    final_solution = set(best_solution)
    
    # Recalculate cover counts
    final_cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in final_solution:
        final_cover_counts[block_t_ids[b_idx]] += 1
        
    # Find uncovered
    uncovered_mask = final_cover_counts == 0
    uncovered_t_ids = np.where(uncovered_mask)[0]
    
    # Greedy repair
    while len(uncovered_t_ids) > 0:
        if time.time() - start_time > time_limit - 2.0:
            break
            
        target_tid = uncovered_t_ids[0]
        
        # Find block covering target_tid that covers most other uncovered
        candidates = t_to_blocks[target_tid]
        best_block = -1
        max_new_cover = -1
        
        for b_idx in candidates:
            if b_idx in final_solution:
                continue
            b_ids = block_t_ids[b_idx]
            # Count how many of these are currently uncovered
            new_cover = np.sum(final_cover_counts[b_ids] == 0)
            
            if new_cover > max_new_cover:
                max_new_cover = new_cover
                best_block = b_idx
        
        if best_block != -1:
            final_solution.add(best_block)
            b_ids = block_t_ids[best_block]
            final_cover_counts[b_ids] += 1
            # Update uncovered_t_ids
            # This is inefficient to update the numpy array every time.
            # But repair phase is short.
            # We can just recompute uncovered_t_ids if it's slow, or update it.
            # Let's just recompute at the end of the loop if we add a block?
            # Or just break if we run out of time.
            # For correctness, we need to ensure we don't pick a block that doesn't cover target_tid.
            # The loop condition checks len(uncovered_t_ids).
            # We should update uncovered_t_ids to remove the newly covered ones.
            
            # Efficient update:
            # new_covered = b_ids[final_cover_counts[b_ids] == 1] # Just those that became 1 from 0
            # Actually, we added 1 to all. So those that were 0 are now 1.
            # We can find which ones were 0 before the update?
            # We didn't store the old state.
            # Let's just recompute uncovered_t_ids. It's O(N).
            # N=11k. Repair steps < 100. 1.1M ops. Fine.
            uncovered_mask = final_cover_counts == 0
            uncovered_t_ids = np.where(uncovered_mask)[0]
        else:
            break
            
    result = [list(all_k_subsets[i]) for i in final_solution]
    return result
