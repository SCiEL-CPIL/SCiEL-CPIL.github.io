<<<<<<< SEARCH
import time
import random
import itertools
=======
import time
import random
import itertools
import math
>>>>>>> REPLACE
<<<<<<< SEARCH
        # Multi-start Local Search
        # Start from the 243 set, but with different random shuffles
        # Also try starting from a greedy set cover if time permits
        
        initial_list = [word_to_idx[c] for c in initial_codewords]
        
        # Run multiple restarts
        restarts = 0
        while time.time() < deadline - 10:
            current_solution = optimize(initial_list)
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                # Store best solution
                best_indices = current_solution
            
            # Try a different start: Greedy Set Cover
            if restarts == 0 and time.time() < deadline - 30:
                # Simple Greedy
                uncovered_mask = (1 << N) - 1
                greedy_indices = []
                # Precompute neighbor masks list for faster access? Already have neighbor_masks.
                
                # To speed up greedy, we can estimate the number of newly covered words.
                # Since N is small, we can just scan.
                while uncovered_mask and time.time() < deadline - 15:
                    best_idx = -1
                    best_gain = 0
                    # Sample a subset of candidates if full scan is too slow?
                    # Full scan is 2187 iterations. Each iteration involves bitwise AND and BIT_COUNT.
                    # Python int bit_count is fast (C implementation).
                    for idx in range(N):
                        if idx in greedy_indices: continue # O(1) if set, O(N) if list. Use set.
                        # We don't have a set for greedy_indices yet. Let's use a boolean array.
                        # Actually, just checking `if idx in greedy_set` where greedy_set is a set.
                        pass
                    
                    # Let's implement a simpler greedy with a set for O(1) lookup
                    greedy_set = set(greedy_indices)
                    for idx in range(N):
                        if idx in greedy_set: continue
                        gain = (neighbor_masks[idx] & uncovered_mask).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = idx
                            if gain == uncovered_mask.bit_count(): break
                            
                    if best_idx == -1 or best_gain == 0: break
                    greedy_indices.append(best_idx)
                    greedy_set.add(best_idx)
                    uncovered_mask &= ~neighbor_masks[best_idx]
                
                if len(greedy_indices) < best_size:
                    # Optimize this greedy solution
                    optimized_greedy = optimize(greedy_indices)
                    if len(optimized_greedy) < best_size:
                        best_size = len(optimized_greedy)
                        best_indices = optimized_greedy
                
                restarts += 1
                continue

            # Random restart: Perturb the initial list
            # Shuffle initial list and maybe add/remove a few random elements?
            # Just shuffling is enough for the removal heuristic to explore different paths.
            
            # If we haven't improved in a while, try a random restart from scratch?
            # Greedy is expensive. Let's stick to optimizing the tetracode seed with different shuffles.
            # The `optimize` function already shuffles internally.
            
            restarts += 1
            
        # Return the best found
        # best_indices is the list of indices
        result = [list(all_words[idx]) for idx in best_indices]
        return result
=======
        # Multi-start Local Search with Simulated Annealing for Escape
        
        initial_list = [word_to_idx[c] for c in initial_codewords]
        
        # Function to calculate cost (number of uncovered words)
        # We use cover_count array. Uncovered words are those with count 0.
        # To make this fast, we maintain cover_count and a count of uncovered words.
        # But cover_count is local to optimize. Let's write a separate SA function or integrate it.
        
        def simulated_annealing(initial_indices, time_limit):
            # State: list of indices
            current_indices = list(initial_indices)
            random.shuffle(current_indices)
            
            # Initialize cover counts
            cover_count = [0] * N
            for idx in current_indices:
                mask = neighbor_masks[idx]
                m = mask
                while m:
                    lb = m & -m
                    idx_nb = lb.bit_length() - 1
                    cover_count[idx_nb] += 1
                    m ^= lb
            
            # Calculate initial uncovered count
            uncovered_count = 0
            for c in cover_count:
                if c == 0:
                    uncovered_count += 1
            
            current_cost = uncovered_count + 1000 * len(current_indices) # Weight size heavily? 
            # Actually, we want to minimize size primarily.
            # If we allow uncovered words, we might get smaller sets that are invalid.
            # The task requires VALID codes. So we must ensure uncovered_count == 0 at the end.
            # SA can explore invalid states to escape local minima in valid states.
            # Cost = alpha * size + beta * uncovered_count
            # We want to find a valid code (uncovered=0) with min size.
            # If we find a valid code, we can try to reduce size further by swapping.
            
            # Let's use Cost = size + 100 * uncovered_count
            # This encourages covering all words, but allows temporary violations to find better structures.
            
            current_cost = len(current_indices) + 100 * uncovered_count
            best_indices = list(current_indices)
            best_cost = current_cost
            best_uncovered = uncovered_count
            
            T = 10.0
            cooling = 0.995
            start_time = time.time()
            
            while time.time() < start_time + time_limit:
                if T < 0.01:
                    break
                
                # Generate Move
                move_type = random.choice(['remove', 'add', 'swap'])
                
                # 1. Remove
                if move_type == 'remove' and len(current_indices) > 0:
                    i = random.randrange(len(current_indices))
                    idx = current_indices[i]
                    
                    # Calculate delta cost
                    # New cover_count = old - neighbors[idx]
                    # Delta size = -1
                    # Delta uncovered = (new_count == 0 for neighbors where old_count == 1)
                    
                    delta_uncovered = 0
                    mask = neighbor_masks[idx]
                    m = mask
                    while m:
                        lb = m & -m
                        idx_nb = lb.bit_length() - 1
                        if cover_count[idx_nb] == 1:
                            delta_uncovered += 1
                        m ^= lb
                    
                    delta_cost = -1 + 100 * delta_uncovered
                    
                    # Accept
                    if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                        # Apply
                        current_indices.pop(i)
                        m = mask
                        while m:
                            lb = m & -m
                            idx_nb = lb.bit_length() - 1
                            cover_count[idx_nb] -= 1
                            m ^= lb
                        
                        uncovered_count += delta_uncovered
                        current_cost += delta_cost
                        
                        if current_cost < best_cost or (current_uncovered == 0 and len(current_indices) < len(best_indices)):
                            # Note: best_cost includes penalty. We should track valid best separately.
                            pass
                        
                # 2. Add
                elif move_type == 'add':
                    idx = random.randint(0, N - 1)
                    if idx not in current_indices: # O(N) check, acceptable for small N?
                        # Calculate delta cost
                        delta_uncovered = 0
                        mask = neighbor_masks[idx]
                        m = mask
                        while m:
                            lb = m & -m
                            idx_nb = lb.bit_length() - 1
                            if cover_count[idx_nb] == 0:
                                delta_uncovered += 1
                            m ^= lb
                        
                        delta_cost = 1 - 100 * delta_uncovered
                        
                        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                            current_indices.append(idx)
                            m = mask
                            while m:
                                lb = m & -m
                                idx_nb = lb.bit_length() - 1
                                cover_count[idx_nb] += 1
                                m ^= lb
                            
                            uncovered_count -= delta_uncovered
                            current_cost += delta_cost
                            
                # 3. Swap
                elif move_type == 'swap' and len(current_indices) > 0:
                    i = random.randrange(len(current_indices))
                    idx_a = current_indices[i]
                    idx_b = random.randint(0, N - 1)
                    if idx_b != idx_a and idx_b not in current_indices:
                        mask_a = neighbor_masks[idx_a]
                        mask_b = neighbor_masks[idx_b]
                        
                        # Delta size = 0
                        # Delta uncovered:
                        # Words in A but not B that were only covered by A become uncovered (+1)
                        # Words in B but not A that were uncovered become covered (-1)
                        
                        delta_uncovered = 0
                        
                        # Check words in A
                        m = mask_a
                        while m:
                            lb = m & -m
                            idx_nb = lb.bit_length() - 1
                            # If not in B
                            if not (mask_b & lb):
                                if cover_count[idx_nb] == 1:
                                    delta_uncovered += 1
                            m ^= lb
                        
                        # Check words in B
                        m = mask_b
                        while m:
                            lb = m & -m
                            idx_nb = lb.bit_length() - 1
                            # If not in A
                            if not (mask_a & lb):
                                if cover_count[idx_nb] == 0:
                                    delta_uncovered -= 1
                            m ^= lb
                        
                        delta_cost = 100 * delta_uncovered
                        
                        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                            # Apply
                            current_indices[i] = idx_b
                            
                            # Update cover counts
                            # Subtract A
                            m = mask_a
                            while m:
                                lb = m & -m
                                idx_nb = lb.bit_length() - 1
                                cover_count[idx_nb] -= 1
                                m ^= lb
                            
                            # Add B
                            m = mask_b
                            while m:
                                lb = m & -m
                                idx_nb = lb.bit_length() - 1
                                cover_count[idx_nb] += 1
                                m ^= lb
                            
                            uncovered_count += delta_uncovered
                            current_cost += delta_cost
                
                # Update Best
                if uncovered_count == 0:
                    if len(current_indices) < len(best_indices):
                        best_indices = list(current_indices)
                        best_cost = current_cost
                        best_uncovered = 0
                else:
                    # If we have a valid best, we might still accept invalid states if they have lower "potential" size?
                    # No, we only track valid bests for the final answer.
                    # But we can use cost to guide search.
                    pass

                T *= cooling
                
                # Periodic local optimization (greedy removal) to clean up
                if random.random() < 0.1:
                    # Simple greedy removal pass
                    improved = True
                    while improved:
                        improved = False
                        random.shuffle(current_indices)
                        for i in range(len(current_indices)):
                            idx = current_indices[i]
                            mask = neighbor_masks[idx]
                            m = mask
                            redundant = True
                            while m:
                                lb = m & -m
                                idx_nb = lb.bit_length() - 1
                                if cover_count[idx_nb] <= 1:
                                    redundant = False
                                    break
                                m ^= lb
                            
                            if redundant:
                                m = mask
                                while m:
                                    lb = m & -m
                                    idx_nb = lb.bit_length() - 1
                                    cover_count[idx_nb] -= 1
                                    m ^= lb
                                
                                current_indices.pop(i)
                                uncovered_count += 0 # Should remain 0 if it was valid, or increase if it was invalid
                                # Recalculate uncovered count is expensive. 
                                # If we only do this on valid states, uncovered_count stays 0.
                                # If invalid, it might change.
                                # Let's just recalculate uncovered_count if it's small or if we want to be safe.
                                # For now, assume we only do this when close to valid.
                                improved = True
                                if len(current_indices) < len(best_indices) and uncovered_count == 0:
                                    best_indices = list(current_indices)
                                break # Restart loop after pop
                    
                    # Recalculate uncovered count to be safe
                    if uncovered_count != 0:
                        uncovered_count = 0
                        for c in cover_count:
                            if c == 0:
                                uncovered_count += 1
                        current_cost = len(current_indices) + 100 * uncovered_count
                        if uncovered_count == 0 and len(current_indices) < len(best_indices):
                            best_indices = list(current_indices)
                            best_cost = current_cost

            return best_indices

        # Run Simulated Annealing
        sa_time_limit = 30.0
        if time.time() < deadline - sa_time_limit - 5:
            sa_result = simulated_annealing(initial_list, sa_time_limit)
            if len(sa_result) < best_size:
                best_size = len(sa_result)
                best_indices = sa_result
        
        # Run a few more local search restarts with the best found so far
        restarts = 0
        while time.time() < deadline - 5:
            current_solution = optimize(best_indices)
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_indices = current_solution
            
            restarts += 1
            if restarts > 5: break # Limit restarts to avoid infinite loop if no improvement

        # Return the best found
        result = [list(all_words[idx]) for idx in best_indices]
        return result
>>>>>>> REPLACE
