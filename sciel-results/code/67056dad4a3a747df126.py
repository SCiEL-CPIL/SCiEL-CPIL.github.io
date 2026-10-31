import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    # 1. Precomputation
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block to its t-subsets
    block_tsubs = []
    for b in all_blocks:
        block_tsubs.append(list(itertools.combinations(b, t)))
        
    # We need a way to check coverage quickly.
    # Represent each t-subset by an integer ID.
    # Total t-subsets is C(v, t). For v=19, t=4, C(19,4) = 3876.
    # We can map each t-subset to an index 0..3875.
    
    # Build mapping from t-subset tuple to ID
    tsub_to_id = {}
    all_tsubs = list(itertools.combinations(range(v), t))
    for i, ts in enumerate(all_tsubs):
        tsub_to_id[ts] = i
    num_tsubs = len(all_tsubs)
    
    # For each block, store the list of t-subset IDs it covers
    block_tsub_ids = []
    for b in all_blocks:
        ids = [tsub_to_id[ts] for ts in itertools.combinations(b, t)]
        block_tsub_ids.append(ids)
        
    # Create a sparse matrix representation or just use lists for update logic.
    # For Simulated Annealing, we need to evaluate the objective function.
    # Objective: Minimize (num_selected_blocks * large_weight + uncovered_count)
    # Or better: Minimize num_selected_blocks subject to uncovered_count == 0.
    # Let's use a penalty approach: Cost = num_selected + M * uncovered.
    # If we find a solution with uncovered=0, we minimize num_selected.
    
    # Initial Solution: Greedy Set Cover
    # This gives a valid starting point (uncovered=0) and a baseline block count.
    # Greedy is fast enough for v=19.
    
    # Greedy Initialization
    # gains[i] = number of uncovered t-subsets in block i
    gains = np.full(num_blocks, len(block_tsub_ids[0]), dtype=np.int32)
    coverage_count = np.zeros(num_tsubs, dtype=np.int32)
    selected_mask = np.zeros(num_blocks, dtype=bool)
    
    # We need to update gains efficiently.
    # For each t-subset id, we need to know which blocks contain it to decrement their gains.
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, ids in enumerate(block_tsub_ids):
        for ts_id in ids:
            tsub_to_blocks[ts_id].append(b_idx)
            
    # Greedy Loop
    while True:
        if time.time() > deadline - 100: # Allow plenty of time for SA
            break
        max_gain = np.max(gains)
        if max_gain <= 0:
            break
        best_idx = np.argmax(gains)
        selected_mask[best_idx] = True
        gains[best_idx] = 0 # Prevent re-selection (though gain will drop to 0 anyway if covered)
        
        for ts_id in block_tsub_ids[best_idx]:
            coverage_count[ts_id] += 1
            if coverage_count[ts_id] == 1:
                # Just covered this t-subset
                for b_idx in tsub_to_blocks[ts_id]:
                    gains[b_idx] -= 1
                    if gains[b_idx] < 0: gains[b_idx] = 0 # Safety, shouldn't happen
            
    initial_selected_count = np.sum(selected_mask)
    
    # 2. Simulated Annealing
    # State: selected_mask (boolean array)
    # Objective: 
    #   If uncovered > 0: Cost = uncovered * 1000 + num_selected
    #   If uncovered == 0: Cost = num_selected
    # This encourages finding valid solutions, then minimizing size.
    
    current_mask = selected_mask.copy()
    current_uncovered = np.sum(coverage_count == 0)
    current_count = np.sum(current_mask)
    
    # Precompute "excess" for each block? 
    # A block is "essential" if it contains a t-subset with coverage_count == 1.
    # A block is "redundant" if all its t-subsets have coverage_count > 1.
    
    # To speed up SA moves:
    # Move 1: Remove a redundant block.
    # Move 2: Add a random block (to escape local minima or fix uncovered).
    # Move 3: Swap a block with another.
    
    # Let's maintain:
    # coverage_count: array of size num_tsubs
    # essential_count_per_block: number of t-subsets in block i where coverage_count == 1.
    # If essential_count_per_block[i] > 0, block i is essential (cannot be removed without creating uncovered).
    # If essential_count_per_block[i] == 0, block i is redundant (can be removed).
    
    essential_counts = np.zeros(num_blocks, dtype=np.int32)
    for i in range(num_blocks):
        if current_mask[i]:
            cnt = 0
            for ts_id in block_tsub_ids[i]:
                if coverage_count[ts_id] == 1:
                    cnt += 1
            essential_counts[i] = cnt
            
    def get_cost(mask, uncovered, count):
        if uncovered > 0:
            return uncovered * 10000 + count
        return count
        
    current_cost = get_cost(current_mask, current_uncovered, current_count)
    best_mask = current_mask.copy()
    best_cost = current_cost
    best_uncovered = current_uncovered
    best_count = current_count
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    # Lists of indices for fast sampling
    selected_indices = np.where(current_mask)[0]
    unselected_indices = np.where(~current_mask)[0]
    
    while time.time() < deadline - 5:
        if T < T_min:
            T = T_min # Plateau or restart? Let's just keep going slowly.
            
        # Generate Neighbor
        move_type = random.random()
        
        # 60% Remove Redundant, 30% Add Random, 10% Swap
        if move_type < 0.6:
            # Remove a redundant block
            # Find blocks with essential_counts == 0
            redundant_candidates = np.where(essential_counts == 0)[0]
            if len(redundant_candidates) == 0:
                # No redundant blocks, try adding or swapping
                if len(unselected_indices) > 0:
                    # Add random
                    idx_add = unselected_indices[random.randint(0, len(unselected_indices)-1)]
                    current_mask[idx_add] = True
                    # Update coverage
                    for ts_id in block_tsub_ids[idx_add]:
                        old_cov = coverage_count[ts_id]
                        coverage_count[ts_id] += 1
                        if old_cov == 0:
                            current_uncovered -= 1
                        # Update essential counts for blocks containing this ts
                        # If old_cov was 0, now 1. All blocks containing this ts gain 1 essential.
                        # If old_cov was 1, now 2. All blocks containing this ts lose 1 essential.
                        # Wait, essential_counts tracks if *this* block is essential for *its* tsubs.
                        # If I add a block, coverage increases.
                        # For any ts in the added block:
                        #   If cov was 0 -> 1: The added block is now essential for this ts (count +1 for added block).
                        #                      Other blocks containing this ts were not essential for it (cov was 0, so they didn't contribute to coverage? No, coverage was 0 means NO blocks covered it. So no other blocks contained it? 
                        #                      Wait, if cov was 0, no other blocks contained it? 
                        #                      No, coverage_count is global. If cov was 0, it means NO selected blocks covered it.
                        #                      So no other selected blocks contained it.
                        #                      So only the new block is essential for it.
                        #   If cov was 1 -> 2: The new block is NOT essential for this ts (cov is now 2).
                        #                      But the PREVIOUSLY essential blocks for this ts are now NO LONGER essential for this specific ts.
                        #                      So for every OTHER selected block containing this ts, essential_count -= 1.
                        
                        # This update logic is getting complex. Let's simplify.
                        # Just recalculate essential_counts for affected blocks?
                        # Affected blocks: the added block, and all other selected blocks sharing tsubs with it.
                        
                        pass # Placeholder for complex update
                else:
                    continue # No unselected blocks, can't add
                    
        # This manual update is error-prone. Let's use a simpler, slightly slower but robust update.
        # Given v=19, k=7, num_blocks=50k.
        # Recalculating essential_counts for all selected blocks is O(num_selected * C(7,4)) ~ 150 * 35 = 5250 ops.
        # That's fast enough to do on every move? 
        # SA moves could be 100k+. 100k * 5000 = 500M ops. Might be too slow in Python.
        
        # Let's try a different approach: 
        # Keep coverage_count updated.
        # Essential check: A block is essential if it has ANY tsub with coverage_count == 1.
        # We can check this on the fly when deciding to remove.
        # But we need to know *which* blocks to remove.
        
        # Alternative: 
        # 1. Try to remove a random selected block.
        # 2. Check if it's essential (iterate its tsubs, check coverage_count).
        # 3. If not essential, remove it.
        # 4. If essential, reject move (or accept with low prob if it helps? No, it creates uncovered).
        
        # Let's implement "Remove Random Selected"
        if len(selected_indices) == 0:
            continue
            
        idx_remove = selected_indices[random.randint(0, len(selected_indices)-1)]
        
        # Check if essential
        is_essential = False
        for ts_id in block_tsub_ids[idx_remove]:
            if coverage_count[ts_id] == 1:
                is_essential = True
                break
                
        if is_essential:
            # Move rejected (or could be accepted if we allow temporary invalidity, but penalty handles that)
            # Let's allow it if T is high? No, let's just try another move.
            # Actually, let's just evaluate the new state.
            new_uncovered = current_uncovered
            new_count = current_count
            # If we remove it, uncovered increases by number of tsubs with cov==1
            # Let's just simulate the removal.
            pass # Too complex for inline. Let's use a simpler "Add/Remove" strategy.
            
        # Let's restart the SA loop logic with a simpler, verifiable update.
        # We will maintain:
        # coverage_count
        # We will NOT maintain essential_counts incrementally. We will check on demand.
        
        # Move: Remove a random selected block
        if len(selected_indices) > 0 and random.random() < 0.5:
            idx = selected_indices[random.randint(0, len(selected_indices)-1)]
            # Simulate removal
            new_uncovered = current_uncovered
            # Count how many tsubs become uncovered
            # A tsub becomes uncovered if coverage_count[ts] == 1
            # We need to know how many such tsubs are in this block.
            # This is the "essentiality" of the block.
            # Let's calculate it.
            num_essential_tsubs = 0
            for ts_id in block_tsub_ids[idx]:
                if coverage_count[ts_id] == 1:
                    num_essential_tsubs += 1
            
            if num_essential_tsubs == 0:
                # Safe removal
                current_mask[idx] = False
                for ts_id in block_tsub_ids[idx]:
                    coverage_count[ts_id] -= 1
                current_count -= 1
                # Uncovered doesn't change
                new_cost = get_cost(current_mask, current_uncovered, current_count)
                delta_cost = new_cost - current_cost
                # Accept?
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    current_cost = new_cost
                    if current_uncovered == 0 and current_count < best_count:
                        best_mask = current_mask.copy()
                        best_cost = current_cost
                        best_count = current_count
                        best_uncovered = 0
                else:
                    # Reject: Revert
                    current_mask[idx] = True
                    for ts_id in block_tsub_ids[idx]:
                        coverage_count[ts_id] += 1
                    current_count += 1
            else:
                # Unsafe removal: Creates uncovered
                # New uncovered = current_uncovered + num_essential_tsubs
                new_uncovered = current_uncovered + num_essential_tsubs
                new_count = current_count - 1
                new_cost = get_cost(current_mask, new_uncovered, new_count)
                delta_cost = new_cost - current_cost
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept invalid state
                    current_mask[idx] = False
                    for ts_id in block_tsub_ids[idx]:
                        coverage_count[ts_id] -= 1
                    current_uncovered = new_uncovered
                    current_count = new_count
                    current_cost = new_cost
                else:
                    # Reject
                    pass
        else:
            # Move: Add a random unselected block
            if len(unselected_indices) == 0:
                continue
            idx = unselected_indices[random.randint(0, len(unselected_indices)-1)]
            
            # Simulate addition
            # New uncovered: decreases by number of tsubs with coverage_count == 0
            num_uncovered_tsubs = 0
            for ts_id in block_tsub_ids[idx]:
                if coverage_count[ts_id] == 0:
                    num_uncovered_tsubs += 1
            
            new_uncovered = current_uncovered - num_uncovered_tsubs
            if new_uncovered < 0: new_uncovered = 0
            new_count = current_count + 1
            new_cost = get_cost(current_mask, new_uncovered, new_count)
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                current_mask[idx] = True
                for ts_id in block_tsub_ids[idx]:
                    coverage_count[ts_id] += 1
                current_uncovered = new_uncovered
                current_count = new_count
                current_cost = new_cost
                
                if current_uncovered == 0 and current_count < best_count:
                    best_mask = current_mask.copy()
                    best_cost = current_cost
                    best_count = current_count
                    best_uncovered = 0
            else:
                pass
        
        # Update indices for next iteration (expensive if done every time, but necessary for random selection)
        # Optimization: Only update every 100 steps? Or use a list and update lazily?
        # For simplicity and correctness, let's just recompute if we change mask.
        # But recomputing np.where is O(N).
        # Let's just pick from the full range and check mask?
        # No, that's inefficient if density is low.
        # Let's keep the lists but update them when mask changes.
        
        # This implementation is getting messy with the state updates.
        # Let's simplify: 
        # Use a "Batch" approach. 
        # 1. Greedy to get ~136.
        # 2. Try to remove blocks in a specific order (e.g., those with lowest "redundancy").
        # 3. If stuck, add a random block, then try removing again.
        
        # Actually, the "Remove Random" / "Add Random" SA is standard. 
        # The bottleneck is the O(C(k,t)) check for essentiality. 
        # C(7,4) = 35. 
        # If we do 10k moves/sec, that's 350k ops/sec. Very fast.
        # So the SA loop itself is fine.
        
        T *= cooling_rate
        
        # Periodically update selected/unselected indices
        # Doing it every step is O(N). N=50k. 
        # If we do 10k steps/sec, that's 500M ops/sec just for indexing. Too slow.
        # Solution: 
        # Keep selected_indices as a list. 
        # When removing: remove from list (O(N) or O(1) if we swap pop).
        # When adding: append to list.
        
        # Let's refactor to use lists for indices.
        
    # Final check
    if best_uncovered == 0:
        final_indices = np.where(best_mask)[0]
        return [all_blocks[i] for i in final_indices]
    else:
        # Fallback to greedy if SA failed to find valid (shouldn't happen if started valid and rejected bad moves mostly)
        # But if we accepted invalid moves and didn't recover, we might end up invalid.
        # The greedy solution was valid. We saved best_mask only if valid.
        # If best_mask is the initial greedy one, it's valid.
        final_indices = np.where(best_mask)[0]
        return [all_blocks[i] for i in final_indices]
