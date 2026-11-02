<<<<<<< SEARCH
    # Local Search Loop
    random.seed(42)
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # 1. Try to remove redundant words
        removed_any = False
        for i in range(len(selected) - 1, -1, -1):
            idx = selected[i]
            # Check if removing idx leaves any element with count 0
            # temp_counts = counts - balls_matrix[idx]
            # if np.any(temp_counts == 0): continue
            # else: remove
            
            # Vectorized check:
            # We need to know if counts[j] > 1 for all j in balls[idx]
            # If yes, idx is redundant.
            if np.all(counts[balls[idx]] > 1):
                counts[balls[idx]] -= 1
                selected.pop(i)
                removed_any = True
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_selected = list(selected)
        
        if not removed_any:
            # 2. Try a swap
            if len(selected) == 0:
                break
            
            # Pick a random word to remove
            i = random.randint(0, len(selected) - 1)
            idx_to_remove = selected[i]
            
            # Calculate deficit if we remove it
            # deficit[j] = 1 if counts[j] == 1 and j is in balls[idx_to_remove]
            # Actually, if we remove idx, new_counts = counts - balls_matrix[idx]
            # We need to find v such that new_counts + balls_matrix[v] > 0 everywhere.
            # This is equivalent to balls_matrix[v] covering all j where new_counts[j] == 0.
            
            new_counts = counts - balls_matrix[idx_to_remove]
            missing_indices = np.where(new_counts == 0)[0]
            
            if len(missing_indices) == 0:
                # We can just remove it (should have been caught by redundancy check, 
                # but maybe counts changed? No, counts is consistent.)
                # This case is handled above.
                pass
            else:
                # Find v that covers all missing_indices
                # covers_deficit = balls_matrix[:, missing_indices]
                # valid_v = np.all(covers_deficit, axis=1)
                
                # To speed up, we can check candidates.
                # But checking all N is 2187 * len(missing) ops.
                # len(missing) is small (size of ball of idx, max 19).
                # So 2187 * 19 ~ 40k ops. Very fast.
                
                # We want to pick a v not in selected.
                # And ideally one that minimizes "waste" or allows further removal.
                
                # Let's just find any valid v.
                # valid_v_mask = np.all(balls_matrix[:, missing_indices], axis=1)
                # valid_v_mask[selected] = False # Exclude already selected
                
                # For efficiency, we can just pick a random subset of candidates or all.
                # Let's compute valid_v_mask
                valid_v_mask = np.all(balls_matrix[:, missing_indices], axis=1)
                
                # Exclude selected
                for sel_idx in selected:
                    valid_v_mask[sel_idx] = False
                
                valid_indices = np.where(valid_v_mask)[0]
                
                if len(valid_indices) > 0:
                    # Pick one randomly
                    new_idx = valid_indices[random.randint(0, len(valid_indices) - 1)]
                    
                    # Perform swap
                    counts[balls[idx_to_remove]] -= 1
                    counts[balls[new_idx]] += 1
                    selected[i] = new_idx
                    
                    # After swap, try to remove redundancies again immediately
                    # (Loop back to step 1)
                    continue # Restart the redundancy check phase

        # If we get here, no removal and no valid swap found in this iteration?
        # We can try a random restart or perturbation if stuck, but for now, 
        # the swap logic above attempts to find a valid swap.
        # If no valid swap exists, we are at a local minimum for swap moves.
        
=======
    # Local Search Loop
    random.seed(42)
    stuck_iterations = 0
    
    def try_remove_redundant():
        """Greedy removal of redundant words, prioritizing those that unlock others."""
        nonlocal counts, selected
        removed_any = False
        # Sort selected by "impact" of removal? 
        # A word is redundant if all its neighbors are covered >= 2 times.
        # If we remove it, counts decrease.
        # We want to remove words that are "least useful" for maintaining coverage of critical nodes.
        # Actually, standard greedy is: remove any redundant. 
        # To improve, we can remove the one that minimizes the number of nodes that become "critical" (count=1) 
        # or maximizes the number of nodes that remain "safe" (count>1).
        
        # For speed, we just iterate and remove if possible.
        # We do a pass.
        i = 0
        while i < len(selected):
            idx = selected[i]
            # Check redundancy
            if np.all(counts[balls[idx]] > 1):
                # It is redundant.
                # Before removing, let's see if we can improve by choosing a different one?
                # Simple strategy: remove it.
                counts[balls[idx]] -= 1
                selected.pop(i)
                removed_any = True
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_selected = list(selected)
            else:
                i += 1
        return removed_any

    def find_swap():
        """Find a 1-swap that maintains validity."""
        nonlocal counts, selected
        if not selected:
            return False
        
        # Try to remove a random word and find a replacement
        # To be efficient, pick a word that is "least redundant" or random.
        # Random is fine for diversity.
        i = random.randint(0, len(selected) - 1)
        idx_to_remove = selected[i]
        
        new_counts = counts - balls_matrix[idx_to_remove]
        missing_indices = np.where(new_counts == 0)[0]
        
        if len(missing_indices) == 0:
            # This should have been caught by try_remove_redundant, 
            # but counts might be stale if we don't update? No, counts is updated.
            # If missing is 0, it IS redundant.
            counts[balls[idx_to_remove]] -= 1
            selected.pop(i)
            if len(selected) < best_count:
                best_count = len(selected)
                best_selected = list(selected)
            return True
        
        # Find replacement v
        # v must cover all missing_indices
        valid_v_mask = np.all(balls_matrix[:, missing_indices], axis=1)
        
        # Exclude selected
        for sel_idx in selected:
            valid_v_mask[sel_idx] = False
            
        valid_indices = np.where(valid_v_mask)[0]
        
        if len(valid_indices) > 0:
            # Choose one that maximizes "overlap" with existing coverage to keep counts high?
            # Or just random.
            # Let's pick the one that covers the most *already covered* nodes to boost safety?
            # Actually, we want to keep the solution valid.
            # Random is a good heuristic to explore.
            new_idx = valid_indices[random.randint(0, len(valid_indices) - 1)]
            
            counts[balls[idx_to_remove]] -= 1
            counts[balls[new_idx]] += 1
            selected[i] = new_idx
            return True
            
        return False

    def find_2_swap():
        """Find a 2-swap: remove 2, add 1 or 2. (Simplified: remove 2, add 2)"""
        nonlocal counts, selected
        if len(selected) < 2:
            return False
            
        # Pick 2 random words to remove
        i1, i2 = random.sample(range(len(selected)), 2)
        idx1 = selected[i1]
        idx2 = selected[i2]
        
        # Ensure we don't remove the same one
        if idx1 == idx2: return False
        
        new_counts = counts - balls_matrix[idx1] - balls_matrix[idx2]
        missing_indices = np.where(new_counts == 0)[0]
        
        if len(missing_indices) == 0:
            # Both were redundant? Unlikely if we just did redundancy pass, 
            # but possible if they were mutually redundant.
            counts[balls[idx1]] -= 1
            counts[balls[idx2]] -= 1
            # Remove from selected (careful with indices)
            if i1 > i2: i1, i2 = i2, i1
            selected.pop(i1)
            selected.pop(i2)
            if len(selected) < best_count:
                best_count = len(selected)
                best_selected = list(selected)
            return True
            
        # We need to add words to cover missing_indices.
        # If len(missing) is small, maybe 1 word covers all?
        # Check if 1 word can cover all missing
        valid_v_mask = np.all(balls_matrix[:, missing_indices], axis=1)
        for sel_idx in selected:
            valid_v_mask[sel_idx] = False
        valid_indices = np.where(valid_v_mask)[0]
        
        if len(valid_indices) > 0:
            # Add 1 word, remove 2. Net -1.
            new_idx = valid_indices[random.randint(0, len(valid_indices) - 1)]
            
            counts[balls[idx1]] -= 1
            counts[balls[idx2]] -= 1
            counts[balls[new_idx]] += 1
            
            if i1 > i2: i1, i2 = i2, i1
            selected.pop(i1)
            selected.pop(i2)
            selected.append(new_idx)
            
            if len(selected) < best_count:
                best_count = len(selected)
                best_selected = list(selected)
            return True
            
        # If 1 word isn't enough, try 2 words?
        # This is expensive. We'll skip for now and rely on 1-swaps and redundancy.
        # A 2-in-2-out swap maintains size, which doesn't help minimize, 
        # unless it enables subsequent removals.
        # We'll stick to moves that reduce size or maintain size for perturbation.
        
        return False

    for iteration in range(20000):
        if time.time() > deadline:
            break
            
        # 1. Aggressive Redundancy Removal
        if try_remove_redundant():
            stuck_iterations = 0
            continue
            
        # 2. Try 1-Swap (which might enable further removals)
        if find_swap():
            stuck_iterations = 0
            # After a swap, try removal again immediately
            if try_remove_redundant():
                continue
            continue
            
        # 3. Try 2-Swap (Remove 2, Add 1)
        if find_2_swap():
            stuck_iterations = 0
            continue
            
        # 4. Perturbation if stuck
        stuck_iterations += 1
        if stuck_iterations > 50:
            # Perform a random perturbation:
            # Remove k random words (that are not critical? or just random and fix later)
            # Better: Add k random words, then remove redundancies.
            # This increases size temporarily, but might find a better configuration.
            
            k = random.randint(1, 3)
            added_indices = []
            for _ in range(k):
                # Pick a random word not in selected
                # To speed up, pick from a random subset
                # Or just random from all
                rand_idx = random.randint(0, N - 1)
                # Ensure not in selected (simple check)
                if rand_idx in selected:
                    continue
                selected.append(rand_idx)
                counts[balls[rand_idx]] += 1
                added_indices.append(rand_idx)
                
            stuck_iterations = 0
            # Now clean up
            if try_remove_redundant():
                continue
>>>>>>> REPLACE

I also need to ensure `best_count` and `best_selected` are updated correctly inside the nested functions or passed properly. In Python, `nonlocal` works for variables in the enclosing function scope. `counts` and `selected` are lists/arrays, so modifying them in place works. `best_count` is an integer, so `nonlocal` is required.

One detail: `find_swap` and `find_2_swap` modify `selected` and `counts`. If they succeed, we `continue` the main loop, which restarts the redundancy check. This is a standard approach.

Let's double check the `nonlocal` declarations.
In `construct`:
`counts` is a numpy array.
`selected` is a list.
`best_count` is an int.
`best_selected` is a list.

Inside `try_remove_redundant`:
`nonlocal counts, selected, best_count, best_selected`
Inside `find_swap`:
`nonlocal counts, selected, best_count, best_selected`
Inside `find_2_swap`:
`nonlocal counts, selected, best_count, best_selected`

The edit above includes `nonlocal counts, selected` but I should add `best_count, best_selected` to be safe if I want to update them inside. Actually, in `try_remove_redundant` I do:
`if len(selected) < best_count: best_count = ...`
So I need `nonlocal best_count, best_selected` there too.

Let's refine the edit to include all necessary nonlocals.

