import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    fixed = [1, 2, 5, 11]
    fixed_set = set(fixed)
    max_val = 284
    max_sum = 2 * max_val  # 568
    
    def get_initial_mask(elems):
        mask = 0
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                mask |= (1 << (elems[i] + elems[j]))
        return mask

    def is_valid(x, mask, elems):
        if (mask >> (x + x)) & 1:
            return False
        for y in elems:
            if (mask >> (x + y)) & 1:
                return False
        return True

    def get_new_mask(x, mask, elems):
        new_mask = mask | (1 << (x + x))
        for y in elems:
            new_mask |= (1 << (x + y))
        return new_mask

    def greedy_build(seed):
        random.seed(seed)
        current_elems = list(fixed)
        current_mask = get_initial_mask(current_elems)
        
        candidates = [x for x in range(1, max_val + 1) if x not in fixed_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid(x, current_mask, current_elems):
                current_elems.append(x)
                current_mask = get_new_mask(x, current_mask, current_elems[:-1])
                
        return current_elems, current_mask

    def count_valid_partners(x, mask, elems):
        """Count how many elements in elems can be paired with x without collision."""
        count = 0
        for y in elems:
            if not ((mask >> (x + y)) & 1):
                count += 1
        return count

    def backtrack_solve(current_elems, current_mask, seed, max_depth, time_limit):
        """
        Randomized backtracking with forward checking and dynamic variable ordering.
        Tries to add more elements to current_elems.
        """
        random.seed(seed)
        best_elems = list(current_elems)
        best_mask = current_mask
        best_size = len(current_elems)
        
        # Deadline for this backtracking attempt
        bt_deadline = time.time() + time_limit
        
        def _backtrack(elems, mask, depth):
            nonlocal best_elems, best_mask, best_size, bt_deadline
            
            if time.time() > bt_deadline:
                return
            if depth >= max_depth:
                return
            
            if len(elems) > best_size:
                best_size = len(elems)
                best_elems = list(elems)
                best_mask = mask
            
            # Get candidates: elements not in elems, not in fixed (already in elems), 
            # and check if they can be added
            in_set = set(elems)
            candidates = []
            for x in range(1, max_val + 1):
                if x not in in_set and not x in fixed_set:
                    if is_valid(x, mask, elems):
                        # Count valid partners for ordering
                        partners = count_valid_partners(x, mask, elems)
                        candidates.append((partners, x))
            
            if not candidates:
                return
            
            # Dynamic variable ordering: choose candidate with fewest remaining valid partners
            # This is the "minimum remaining values" heuristic
            candidates.sort(key=lambda t: t[0])
            
            # Take a random subset to add diversity
            # Shuffle the sorted list slightly
            k = min(len(candidates), 5)
            chosen = random.sample(candidates, k)
            
            for _, x in chosen:
                if time.time() > bt_deadline:
                    return
                
                # Add x
                new_elems = elems + [x]
                new_mask = get_new_mask(x, mask, elems)
                
                _backtrack(new_elems, new_mask, depth + 1)
                
                # If we found a solution with more elements, no need to continue
                if best_size >= max_depth:
                    return
        
        _backtrack(current_elems, current_mask, 0)
        
        return best_elems, best_mask

    def local_search(initial_elems, initial_mask, seed):
        """
        Hybrid local search: try to add elements, if stuck do swaps or kicks,
        and periodically run backtracking to escape local optima.
        """
        random.seed(seed)
        best_elems = list(initial_elems)
        best_mask = initial_mask
        best_size = len(initial_elems)
        
        current_elems = list(initial_elems)
        current_mask = initial_mask
        current_size = len(current_elems)
        
        no_improve_count = 0
        max_no_improve = 500
        bt_counter = 0
        
        while time.time() < deadline - 3:
            if no_improve_count > max_no_improve:
                break
                
            improved = False
            
            # 1. Try to add a valid element
            candidates = [x for x in range(1, max_val + 1) if x not in current_elems]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if is_valid(x, current_mask, current_elems):
                    current_elems.append(x)
                    current_mask = get_new_mask(x, current_mask, current_elems[:-1])
                    current_size += 1
                    added = True
                    improved = True
                    break
            
            if not added:
                # 2. Try a swap: remove one non-fixed, add one
                removable = [y for y in current_elems if y not in fixed_set]
                if removable:
                    y_to_remove = random.choice(removable)
                    old_elems = list(current_elems)
                    old_mask = current_mask
                    current_elems.remove(y_to_remove)
                    current_mask = get_initial_mask(current_elems)
                    current_size -= 1
                    
                    candidates = [x for x in range(1, max_val + 1) if x not in current_elems]
                    random.shuffle(candidates)
                    
                    swapped = False
                    for x in candidates:
                        if is_valid(x, current_mask, current_elems):
                            current_elems.append(x)
                            current_mask = get_new_mask(x, current_mask, current_elems[:-1])
                            current_size += 1
                            swapped = True
                            break
                    
                    if not swapped:
                        current_elems = old_elems
                        current_mask = old_mask
                        current_size += 1
                    else:
                        improved = True
                else:
                    break

            if improved:
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_elems = list(current_elems)
                    best_mask = current_mask
            else:
                no_improve_count += 1
                
                # 3. Periodically run backtracking to escape local optima
                bt_counter += 1
                if bt_counter % 50 == 0 or no_improve_count > 100:
                    # Run a short backtracking attempt
                    bt_time = min(1.0, (deadline - time.time() - 3) / 10)
                    if bt_time > 0.1:
                        bt_elems, bt_mask = backtrack_solve(
                            current_elems, current_mask, 
                            random.randint(0, 1000000), 
                            max_depth=current_size + 5,
                            time_limit=bt_time
                        )
                        if len(bt_elems) > current_size:
                            current_elems = bt_elems
                            current_mask = bt_mask
                            current_size = len(current_elems)
                            improved = True
                            if current_size > best_size:
                                best_size = current_size
                                best_elems = list(current_elems)
                                best_mask = current_mask
                        
                        if not improved:
                            # Kick: remove a random non-fixed element
                            removable = [y for y in current_elems if y not in fixed_set]
                            if removable:
                                y_to_remove = random.choice(removable)
                                current_elems.remove(y_to_remove)
                                current_mask = get_initial_mask(current_elems)
                                current_size -= 1
                            else:
                                break
                else:
                    # Just a kick
                    removable = [y for y in current_elems if y not in fixed_set]
                    if removable:
                        y_to_remove = random.choice(removable)
                        current_elems.remove(y_to_remove)
                        current_mask = get_initial_mask(current_elems)
                        current_size -= 1
                    else:
                        break
                        
        return best_elems, best_mask

    best_overall = list(fixed)
    best_mask_overall = get_initial_mask(best_overall)
    best_size = len(best_overall)
    
    # Run multiple iterations of (Greedy + Local Search)
    num_restarts = 25
    time_per_restart = (deadline - time.time() - 5) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Greedy phase
        elems, mask = greedy_build(i)
        
        if len(elems) > best_size:
            best_size = len(elems)
            best_overall = elems
            best_mask_overall = mask
            
        # Local Search phase with backtracking
        final_elems, final_mask = local_search(elems, mask, 1000 + i)
        
        if len(final_elems) > best_size:
            best_size = len(final_elems)
            best_overall = final_elems
            best_mask_overall = final_mask
            
        # If we've reached 20, we can stop early
        if best_size >= 20:
            break
            
    return best_overall
