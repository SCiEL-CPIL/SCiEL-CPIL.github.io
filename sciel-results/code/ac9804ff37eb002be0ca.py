import itertools
import random
import time
import math
import sys

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
    block_t_ids = []
    for block in all_k_subsets:
        ids = [t_subset_to_id[s] for s in itertools.combinations(block, t)]
        block_t_ids.append(ids)
        
    # For each t-subset ID, store the list of k-subset indices that cover it
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, ids in enumerate(block_t_ids):
        for tid in ids:
            t_to_blocks[tid].append(b_idx)

    # 1. Greedy Construction for Initial Solution
    def greedy_cover():
        uncovered = [0] * num_t_subsets # 0 means uncovered, 1 means covered? No, let's use counts or bools.
        # Actually, for greedy, we just need to know if it's covered.
        is_covered = [False] * num_t_subsets
        selected_blocks = set()
        
        # Precompute for speed: for each block, list of t-ids
        # We already have block_t_ids
        
        while True:
            # Find uncovered t-subset
            # Optimization: Pick a random uncovered one to speed up and diversify
            # But standard greedy picks the one that forces the most constraints?
            # Let's just pick a random uncovered one to start, then find best block for it.
            
            # To avoid scanning all t_subsets every time, we can maintain a list of uncovered indices
            # But updating it is O(N). Scanning is O(N).
            # N = 11628. This is fine.
            
            uncovered_indices = [i for i in range(num_t_subsets) if not is_covered[i]]
            if not uncovered_indices:
                break
                
            # Pick a random uncovered subset
            target_tid = random.choice(uncovered_indices)
            
            # Find block covering target_tid that covers most other uncovered subsets
            candidates = t_to_blocks[target_tid]
            best_block = -1
            max_new_cover = -1
            
            for b_idx in candidates:
                if b_idx in selected_blocks:
                    continue
                # Count how many uncovered t-subsets this block covers
                new_cover = 0
                for tid in block_t_ids[b_idx]:
                    if not is_covered[tid]:
                        new_cover += 1
                if new_cover > max_new_cover:
                    max_new_cover = new_cover
                    best_block = b_idx
            
            if best_block != -1:
                selected_blocks.add(best_block)
                for tid in block_t_ids[best_block]:
                    is_covered[tid] = True
            else:
                # Should not happen
                break
                
        return selected_blocks

    # Get initial solution
    initial_solution = greedy_cover()
    
    if not initial_solution:
        # Fallback if greedy failed (unlikely)
        return [list(all_k_subsets[0])]

    # 2. Simulated Annealing Optimization
    # State: set of active blocks
    active_blocks = set(initial_solution)
    
    # Maintain cover counts
    cover_counts = [0] * num_t_subsets
    for b_idx in active_blocks:
        for tid in block_t_ids[b_idx]:
            cover_counts[tid] += 1
            
    # Calculate initial uncovered count
    uncovered_count = sum(1 for c in cover_counts if c == 0)
    
    # Parameters for SA
    # Energy = len(active_blocks) + PENALTY * uncovered_count
    # We want to minimize this.
    # Start with high penalty to force validity, then lower it?
    # Or just keep penalty high enough that we rarely stay invalid.
    # Let's use a dynamic penalty or a fixed high one.
    # If we allow invalid states, we must repair at the end.
    
    T = 100.0
    T_min = 1.0
    cooling_rate = 0.995
    
    # We will run for a certain number of steps or time
    steps = 0
    max_steps = 50000 # Adjust based on time
    
    # To speed up "random block not in solution", we can precompute list of all blocks
    # and filter, or just pick random index and check if in set.
    # Since num_k_subsets is ~75k, picking random index is O(1).
    
    best_solution = set(active_blocks)
    best_uncovered = uncovered_count
    best_size = len(active_blocks)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Choose move type
        # 50% Delete, 20% Insert, 30% Swap?
        # Or just Delete/Insert.
        move_type = random.random()
        
        if move_type < 0.5:
            # Try Delete
            if not active_blocks:
                continue
            b_idx = random.choice(list(active_blocks))
            
            # Calculate delta
            # Removing b_idx
            # New cover counts would be cover_counts[tid] - 1
            # Uncovered increases if cover_counts[tid] == 1
            delta_uncovered = 0
            for tid in block_t_ids[b_idx]:
                if cover_counts[tid] == 1:
                    delta_uncovered += 1
            
            # Energy change:
            # Size decreases by 1
            # Uncovered increases by delta_uncovered
            # Delta E = -1 + PENALTY * delta_uncovered
            # We want to accept if Delta E < 0 or with prob exp(-Delta E / T)
            
            # Let's fix PENALTY to be large, e.g., 1000
            # So if delta_uncovered > 0, it's very costly.
            # If delta_uncovered == 0, Delta E = -1, always accept.
            
            penalty = 1000
            delta_E = -1 + penalty * delta_uncovered
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                active_blocks.remove(b_idx)
                for tid in block_t_ids[b_idx]:
                    cover_counts[tid] -= 1
                uncovered_count += delta_uncovered
                
                # Check if better
                if uncovered_count == 0 and len(active_blocks) < best_size:
                    best_size = len(active_blocks)
                    best_solution = set(active_blocks)
                    best_uncovered = 0
                    
        elif move_type < 0.8:
            # Try Insert
            # Pick a random block not in active_blocks
            # To avoid infinite loops if solution is full (unlikely), just pick random
            b_idx = random.randint(0, num_k_subsets - 1)
            if b_idx in active_blocks:
                continue
            
            # Calculate delta
            # Adding b_idx
            # Uncovered decreases if cover_counts[tid] == 0
            delta_uncovered = 0
            for tid in block_t_ids[b_idx]:
                if cover_counts[tid] == 0:
                    delta_uncovered += 1
            
            # Energy change:
            # Size increases by 1
            # Uncovered decreases by delta_uncovered
            # Delta E = 1 - PENALTY * delta_uncovered
            
            penalty = 1000
            delta_E = 1 - penalty * delta_uncovered
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                active_blocks.add(b_idx)
                for tid in block_t_ids[b_idx]:
                    cover_counts[tid] += 1
                uncovered_count -= delta_uncovered
                
                if uncovered_count == 0 and len(active_blocks) < best_size:
                    best_size = len(active_blocks)
                    best_solution = set(active_blocks)
                    best_uncovered = 0
                    
        else:
            # Try Swap
            if not active_blocks:
                continue
            b_in = random.choice(list(active_blocks))
            b_out = random.randint(0, num_k_subsets - 1)
            if b_out in active_blocks or b_in == b_out:
                continue
            
            # Calculate delta for removing b_in and adding b_out
            # 1. Remove b_in
            # 2. Add b_out
            
            # We need to combine the effects.
            # It's safer to do it in two steps or calculate net change.
            # Net change in cover_counts[tid] = (1 if tid in b_out else 0) - (1 if tid in b_in else 0)
            
            delta_uncovered = 0
            
            # Process b_in (removal)
            # We need to know the state BEFORE adding b_out.
            # Let's simulate:
            # temp_counts is not needed if we just look at current counts.
            
            # Effect of removing b_in:
            # If cover_counts[tid] == 1, it becomes 0 (uncovered) -> +1 to uncovered
            # If cover_counts[tid] > 1, it becomes >0 (covered) -> 0 change
            
            # Effect of adding b_out:
            # If current_cover[tid] (after removal) == 0, it becomes 1 (covered) -> -1 to uncovered
            # If current_cover[tid] (after removal) > 0, it becomes >0 (covered) -> 0 change
            
            # So we must process removal first, then addition on the modified state.
            # But we don't want to modify the actual state yet.
            
            # Let's compute the net change in "uncovered" status.
            # A subset tid is affected if it is in b_in XOR b_out (or both).
            # If in both: count changes by 0.
            # If in b_in only: count decreases by 1.
            # If in b_out only: count increases by 1.
            
            # We need to check the condition on the "intermediate" or "final" state?
            # Uncovered is defined by count == 0.
            
            # Let's iterate through the union of t-ids in b_in and b_out.
            t_ids_in = set(block_t_ids[b_in])
            t_ids_out = set(block_t_ids[b_out])
            affected_t_ids = t_ids_in.union(t_ids_out)
            
            for tid in affected_t_ids:
                c_old = cover_counts[tid]
                
                in_in = 1 if tid in t_ids_in else 0
                in_out = 1 if tid in t_ids_out else 0
                
                c_new = c_old - in_in + in_out
                
                # Check if status changes from covered to uncovered or vice versa
                if c_old > 0 and c_new == 0:
                    delta_uncovered += 1
                elif c_old == 0 and c_new > 0:
                    delta_uncovered -= 1
            
            # Size change: 0 (remove 1, add 1)
            # Delta E = 0 - PENALTY * delta_uncovered
            # Wait, if delta_uncovered is negative (we covered more), Delta E is negative (good).
            # If delta_uncovered is positive (we uncovered some), Delta E is positive (bad).
            
            penalty = 1000
            delta_E = -penalty * delta_uncovered
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept Swap
                # Remove b_in
                active_blocks.remove(b_in)
                for tid in block_t_ids[b_in]:
                    cover_counts[tid] -= 1
                    
                # Add b_out
                active_blocks.add(b_out)
                for tid in block_t_ids[b_out]:
                    cover_counts[tid] += 1
                    
                uncovered_count += delta_uncovered
                
                if uncovered_count == 0 and len(active_blocks) < best_size:
                    best_size = len(active_blocks)
                    best_solution = set(active_blocks)
                    best_uncovered = 0

        T *= cooling_rate
        steps += 1
        
        # Occasionally, if we are stuck (no improvement for a while), we could restart?
        # For now, just let SA cool down.

    # 3. Repair Phase
    # Ensure the best_solution is valid.
    # If best_solution has uncovered subsets, we need to add blocks to cover them.
    # We can use a simple greedy on the uncovered subsets of best_solution.
    
    final_solution = set(best_solution)
    
    # Recalculate cover counts for final_solution to be sure
    final_cover_counts = [0] * num_t_subsets
    for b_idx in final_solution:
        for tid in block_t_ids[b_idx]:
            final_cover_counts[tid] += 1
            
    # Find uncovered
    uncovered_t_ids = [i for i in range(num_t_subsets) if final_cover_counts[i] == 0]
    
    # Greedy repair
    while uncovered_t_ids:
        if time.time() - start_time > time_limit - 2.0:
            break # Stop if running out of time
            
        target_tid = uncovered_t_ids[0]
        
        # Find block covering target_tid that covers most other uncovered
        candidates = t_to_blocks[target_tid]
        best_block = -1
        max_new_cover = -1
        
        for b_idx in candidates:
            if b_idx in final_solution:
                continue
            new_cover = 0
            for tid in block_t_ids[b_idx]:
                if final_cover_counts[tid] == 0:
                    new_cover += 1
            if new_cover > max_new_cover:
                max_new_cover = new_cover
                best_block = b_idx
        
        if best_block != -1:
            final_solution.add(best_block)
            for tid in block_t_ids[best_block]:
                final_cover_counts[tid] += 1
                if final_cover_counts[tid] == 1 and tid in uncovered_t_ids:
                    uncovered_t_ids.remove(tid)
        else:
            # Should not happen
            break
            
    # Final check
    # If still invalid (shouldn't happen), we might return a valid but larger set?
    # The problem requires validity.
    
    # Convert to list of blocks
    result = [list(all_k_subsets[i]) for i in final_solution]
    return result
