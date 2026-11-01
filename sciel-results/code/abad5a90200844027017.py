import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Helper
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # block_to_ts: for each block, list of t-subset indices it covers
    block_to_ts = []
    for block in all_k_blocks:
        ts_list = []
        for ts in itertools.combinations(block, t):
            ts_list.append(ts_index_map[ts])
        block_to_ts.append(ts_list)

    # ts_to_blocks: for each t-subset, list of block indices covering it
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j, ts_list in enumerate(block_to_ts):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(j)

    # Precompute as numpy arrays for speed
    block_to_ts_arr = [np.array(ts_list, dtype=np.int32) for ts_list in block_to_ts]
    
    # Map block tuple to index for fast lookup if needed
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def greedy_cover(seed=0, diversity=1.0):
        """
        Randomized greedy with diversity parameter.
        diversity=1.0 means pick from top 1 (deterministic).
        diversity > 1.0 means pick from top k with probability proportional to gain.
        """
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = np.max(gain)
            if max_gain <= 0:
                break

            # Find candidates with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            
            # If diversity is high, pick from top few? 
            # For now, just pick random from max gain candidates to break symmetry
            b_idx = candidates[random.randint(0, len(candidates) - 1)]

            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1

            for ts_idx in block_to_ts_arr[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1

        return sol

    def remove_redundant(blocks):
        """
        Remove blocks that are not essential (i.e., removing them doesn't uncover any t-subset).
        Returns a new list of blocks (indices).
        """
        if not blocks:
            return []
        
        selected = set(blocks)
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        for b_idx in selected:
            for ts_idx in block_to_ts_arr[b_idx]:
                cover_count[ts_idx] += 1
        
        essential = set()
        # Identify essential blocks
        for b_idx in selected:
            is_essential = False
            for ts_idx in block_to_ts_arr[b_idx]:
                if cover_count[ts_idx] == 1:
                    is_essential = True
                    break
            if is_essential:
                essential.add(b_idx)
        
        return list(essential)

    def local_search_1_for_1(blocks, time_limit):
        """
        Try to swap one block with another to maintain coverage but potentially
        set up for better future merges or just find a smaller equivalent.
        Actually, 1-for-1 swap only helps if we can find a block that covers 
        the unique parts of the removed block.
        """
        selected = set(blocks)
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected:
            for ts_idx in block_to_ts_arr[b_idx]:
                cover_count[ts_idx] += 1
        
        # Current cost is 0 (valid covering)
        # We want to reduce size.
        # A 1-for-1 swap doesn't reduce size.
        # We need 2-for-1 or 1-for-0 (redundancy).
        
        # Let's implement a "2-for-1" merge search.
        # For each pair of blocks (b1, b2), check if there is a block b3 
        # that covers all t-subsets that are uniquely covered by b1 or b2 
        # (i.e., cover_count[ts] == 1 and ts in b1 or ts in b2).
        
        # This is O(N^2 * K) which might be slow. 
        # N ~ 300. N^2 = 90k. K=56. 90k * 56 = 5M ops. Feasible in C, slow in Python.
        # We will sample pairs.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            selected_list = list(selected)
            random.shuffle(selected_list)
            
            # Sample pairs
            num_pairs_to_check = min(1000, (len(selected_list) * (len(selected_list) - 1)) // 2)
            for _ in range(num_pairs_to_check):
                if time.time() > time_limit:
                    break
                i = random.randint(0, len(selected_list) - 1)
                j = random.randint(0, len(selected_list) - 1)
                if i == j:
                    continue
                b1 = selected_list[i]
                b2 = selected_list[j]
                
                # Find t-subsets uniquely covered by b1 or b2
                unique_ts = []
                for ts_idx in block_to_ts_arr[b1]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)
                for ts_idx in block_to_ts_arr[b2]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)
                
                if not unique_ts:
                    # Both blocks are redundant? Should have been caught by remove_redundant
                    # But maybe they are mutually redundant?
                    # If unique_ts is empty, it means every ts in b1 is covered by >=2 blocks, 
                    # and every ts in b2 is covered by >=2 blocks.
                    # Then we can remove b1 and b2? No, we need to check if removing them leaves any ts uncovered.
                    # If unique_ts is empty, it means no ts is *uniquely* covered by them.
                    # So removing them is safe IF we don't break coverage of other ts.
                    # Wait, if cover_count[ts] >= 2 for all ts in b1 and b2, then removing b1 and b2 
                    # reduces count by 2. If count was 2, it becomes 0 -> uncovered.
                    # So unique_ts should be ts where count==1.
                    # If there are no ts with count==1 in b1 or b2, then b1 and b2 are "safe" to remove 
                    # ONLY IF they don't cover any ts with count==1.
                    # Which is exactly what we checked.
                    # So if unique_ts is empty, we can remove BOTH?
                    # Let's verify: 
                    # If I remove b1 and b2, any ts in b1 had count >= 2 (since not in unique_ts). 
                    # After removal, count >= 1. So still covered.
                    # Same for b2.
                    # So we can remove both!
                    selected.discard(b1)
                    selected.discard(b2)
                    for ts_idx in block_to_ts_arr[b1]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts_arr[b2]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart loop with new state
                
                # Check if there is a single block b3 that covers all unique_ts
                # Intersection of ts_to_blocks for all ts in unique_ts
                # Start with blocks covering first ts
                candidate_blocks = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidate_blocks.intersection_update(ts_to_blocks[ts_idx])
                    if not candidate_blocks:
                        break
                
                if candidate_blocks:
                    # Pick one candidate (prefer one not in selected? or any?)
                    # If b3 is b1 or b2, it's not a merge, it's a replacement.
                    # We want to replace {b1, b2} with {b3}.
                    # So b3 should ideally not be b1 or b2.
                    # If b3 is b1, we replace {b1, b2} with {b1} -> remove b2.
                    # That's a 1-for-0 move, which is good.
                    
                    b3 = random.choice(list(candidate_blocks))
                    
                    # Apply change
                    # Remove b1, b2
                    selected.discard(b1)
                    selected.discard(b2)
                    for ts_idx in block_to_ts_arr[b1]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts_arr[b2]:
                        cover_count[ts_idx] -= 1
                    
                    # Add b3
                    if b3 not in selected:
                        selected.add(b3)
                        for ts_idx in block_to_ts_arr[b3]:
                            cover_count[ts_idx] += 1
                    # If b3 was already in selected, we just removed b1,b2 and added nothing new?
                    # No, we added b3 to selected. If it was there, it stays.
                    # But we decremented counts for b1,b2.
                    # And we incremented for b3.
                    
                    improved = True
                    break # Restart loop

        return list(selected)

    # Main ILS Loop
    best_solution = None
    best_len = float('inf')
    
    # Initial greedy
    init_sol = greedy_cover(1)
    if init_sol:
        init_sol = remove_redundant(init_sol)
        best_solution = init_sol
        best_len = len(init_sol)
        
    current_solution = list(init_sol)
    
    iteration = 0
    while time.time() < deadline - 10.0:
        iteration += 1
        
        # Perturb: Remove random blocks
        num_remove = random.randint(3, 10)
        if len(current_solution) <= num_remove:
            current_solution = list(best_solution)
            continue
            
        to_remove = random.sample(current_solution, num_remove)
        perturbed = [b for b in current_solution if b not in set(to_remove)]
        
        # Repair: Greedy add
        # Build state from perturbed
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in perturbed:
            for ts_idx in block_to_ts_arr[b_idx]:
                cover_count[ts_idx] += 1
        is_uncovered = (cover_count == 0)
        
        # Greedy repair
        gain = np.zeros(num_k_blocks, dtype=np.int32)
        # Calculate gain for all blocks
        for b_idx in range(num_k_blocks):
            if b_idx in perturbed:
                continue
            g = 0
            for ts_idx in block_to_ts_arr[b_idx]:
                if is_uncovered[ts_idx]:
                    g += 1
            gain[b_idx] = g
            
        while np.any(is_uncovered):
            if time.time() > deadline - 5.0:
                break
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
            candidates = np.where(gain == max_gain)[0]
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            perturbed.append(b_idx)
            is_uncovered[block_to_ts_arr[b_idx]] = False
            # Update gains
            for ts_idx in block_to_ts_arr[b_idx]:
                if not is_uncovered[ts_idx]: # Was uncovered, now covered
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if nb_idx not in perturbed and gain[nb_idx] > 0:
                            gain[nb_idx] -= 1
                            if gain[nb_idx] < 0:
                                gain[nb_idx] = 0
                            
        # Local Search: Redundancy + 2-for-1
        repaired = remove_redundant(perturbed)
        repaired = local_search_1_for_1(repaired, deadline - 15.0)
        
        if len(repaired) < best_len:
            best_len = len(repaired)
            best_solution = repaired
            current_solution = list(repaired)
        else:
            # Accept with some probability? Or just keep current if not worse?
            # ILS usually accepts worse solutions to escape local minima.
            # But for minimizing block count, we only care about improving.
            # However, if we don't accept worse, we might get stuck.
            # Let's accept if it's not significantly worse (within 5 blocks) with prob 0.1
            if len(repaired) < best_len + 5 and random.random() < 0.1:
                current_solution = repaired

    # Final cleanup
    final_sol = remove_redundant(best_solution)
    final_sol = local_search_1_for_1(final_sol, deadline - 2.0)
    
    result = [list(all_k_blocks[b_idx]) for b_idx in final_sol]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for block in result:
        tblock = tuple(block)
        if tblock not in seen:
            seen.add(tblock)
            unique_result.append(block)
            
    return unique_result
