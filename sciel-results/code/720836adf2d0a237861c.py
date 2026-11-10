"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Denominators are 2 to 48. Indices 0 to 46 correspond to denominators 2 to 48.
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocals as Fractions for exact verification at the end
    # But for search, we use integer scaling for speed.
    
    # Calculate the LCM of all denominators to scale the problem to integers.
    # LCM(2, ..., 48) is a large number, but manageable in Python integers.
    lcm_val = 1
    for d in denominators:
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)
        
    # Integer values representing 1/d * LCM
    int_recs = [lcm_val // d for d in denominators]
    target = lcm_val # Represents 1.0
    
    # Initial solution from the "seed" insight: indices [1, 2, 3, 4, 35, 42]
    # 1->3, 2->4, 3->5, 4->6, 35->37, 42->44
    initial_indices = [1, 2, 3, 4, 35, 42]
    
    current_indices = set(initial_indices)
    current_sum_int = sum(int_recs[i] for i in current_indices)
    current_deficit_int = target - current_sum_int
    
    best_indices = set(current_indices)
    best_deficit_int = current_deficit_int
    
    # Local Search Strategy:
    # 1. Single Add/Remove (if valid)
    # 2. 2-Swap (remove 1, add 1)
    # 3. 3-Swap (remove 2, add 1) or (remove 1, add 2)
    # We will iterate until no improvement is found or deadline is reached.
    # To handle the "stall" issue, we can also try slightly larger perturbations if stuck.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try single additions
        for i in range(n):
            if i not in current_indices:
                new_sum = current_sum_int + int_recs[i]
                if new_sum < target:
                    new_deficit = target - new_sum
                    if new_deficit < current_deficit_int:
                        current_indices.add(i)
                        current_sum_int = new_sum
                        current_deficit_int = new_deficit
                        improved = True
                        if current_deficit_int < best_deficit_int:
                            best_deficit_int = current_deficit_int
                            best_indices = set(current_indices)
        
        # 2. Try single removals (only if it improves deficit? No, removal increases deficit. 
        #    So we only remove if it allows a better swap later. 
        #    However, simple hill climbing doesn't do "worse" moves. 
        #    We will rely on swaps to improve.)
        
        # 3. Try 2-Swaps (remove one in, add one out)
        in_list = list(current_indices)
        out_list = [i for i in range(n) if i not in current_indices]
        
        for i_in in in_list:
            # Check if time is running out
            if time.time() > deadline - 5.0:
                break
            for i_out in out_list:
                new_sum = current_sum_int - int_recs[i_in] + int_recs[i_out]
                if new_sum < target:
                    new_deficit = target - new_sum
                    if new_deficit < current_deficit_int:
                        current_indices.remove(i_in)
                        current_indices.add(i_out)
                        current_sum_int = new_sum
                        current_deficit_int = new_deficit
                        improved = True
                        if current_deficit_int < best_deficit_int:
                            best_deficit_int = current_deficit_int
                            best_indices = set(current_indices)
                        # Update lists if we changed the set
                        in_list = list(current_indices)
                        out_list = [i for i in range(n) if i not in current_indices]
                        # We break out of inner loop to restart outer? 
                        # Actually, just continue searching with the new state.
                        # But the lists are stale. Let's just break and let the while loop restart.
                        break
            if improved:
                break
        
        if improved:
            continue
            
        # 4. If no 2-swap improved, try 3-Swaps (remove 2, add 1) or (remove 1, add 2)
        # This is more expensive, so we do it only when 2-swaps fail.
        
        # 4a. Remove 1, Add 2
        in_list = list(current_indices)
        out_list = [i for i in range(n) if i not in current_indices]
        
        found_3swap = False
        for i_in in in_list:
            if time.time() > deadline - 2.0:
                break
            for i_out1 in out_list:
                for i_out2 in out_list:
                    if i_out1 == i_out2:
                        continue
                    new_sum = current_sum_int - int_recs[i_in] + int_recs[i_out1] + int_recs[i_out2]
                    if new_sum < target:
                        new_deficit = target - new_sum
                        if new_deficit < current_deficit_int:
                            current_indices.remove(i_in)
                            current_indices.add(i_out1)
                            current_indices.add(i_out2)
                            current_sum_int = new_sum
                            current_deficit_int = new_deficit
                            improved = True
                            found_3swap = True
                            if current_deficit_int < best_deficit_int:
                                best_deficit_int = current_deficit_int
                                best_indices = set(current_indices)
                            break
                if found_3swap:
                    break
            if found_3swap:
                break
        
        if found_3swap:
            continue
            
        # 4b. Remove 2, Add 1
        in_list = list(current_indices)
        out_list = [i for i in range(n) if i not in current_indices]
        
        found_3swap = False
        for i_in1 in in_list:
            if time.time() > deadline - 2.0:
                break
            for i_in2 in in_list:
                if i_in1 >= i_in2:
                    continue
                for i_out in out_list:
                    new_sum = current_sum_int - int_recs[i_in1] - int_recs[i_in2] + int_recs[i_out]
                    if new_sum < target:
                        new_deficit = target - new_sum
                        if new_deficit < current_deficit_int:
                            current_indices.remove(i_in1)
                            current_indices.remove(i_in2)
                            current_indices.add(i_out)
                            current_sum_int = new_sum
                            current_deficit_int = new_deficit
                            improved = True
                            found_3swap = True
                            if current_deficit_int < best_deficit_int:
                                best_deficit_int = current_deficit_int
                                best_indices = set(current_indices)
                            break
                if found_3swap:
                    break
            if found_3swap:
                break
        
        if found_3swap:
            continue
            
        # If no 3-swap improved, we might be stuck in a local minimum.
        # We can try a random perturbation (kick) to escape, but we must ensure we don't lose the best solution.
        # Since we track best_indices, we can be aggressive with current_indices.
        
        # Simple random kick: remove 1 random element, add 1 random element (validity checked)
        # If no valid move is found, try removing 2 and adding 1, etc.
        # Given the time budget, we will just stop if no deterministic improvement is found, 
        # as the 2-swap and 3-swap neighborhoods are quite dense for this size.
        
        # However, to be safe against local optima, let's do a few random restarts of the local search 
        # from the current best if we have time.
        # Actually, the "kick" approach is better.
        
        # Let's try a few random swaps to see if we can find a better starting point for the next iteration.
        # We will do this up to 100 times or until time is up.
        
        kicked = False
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            # Pick a random element to remove
            if not current_indices:
                break
            i_in = list(current_indices)[__import__('random').randrange(len(current_indices))]
            out_list = [i for i in range(n) if i not in current_indices]
            if not out_list:
                break
            i_out = out_list[__import__('random').randrange(len(out_list))]
            
            new_sum = current_sum_int - int_recs[i_in] + int_recs[i_out]
            if new_sum < target:
                # Even if it's worse, we accept it to explore? 
                # No, we only want to improve the best. 
                # But if we are stuck, we need to move. 
                # Let's accept any valid move if it doesn't increase deficit too much? 
                # Or just accept it to break the tie.
                # Actually, if we are stuck in a local minimum, the deficit is fixed. 
                # A random move might lead to a better basin.
                current_indices.remove(i_in)
                current_indices.add(i_out)
                current_sum_int = new_sum
                current_deficit_int = target - current_sum_int
                kicked = True
                break
                
        if not kicked:
            # If we couldn't even make a valid random move, we are likely at a very tight constraint.
            # Break the loop.
            break

    # Verify the best solution with exact Fractions
    best_frac_sum = sum(Fraction(1, denominators[i]) for i in best_indices)
    if best_frac_sum >= 1:
        # Fallback to initial known good solution
        return [1, 2, 3, 4, 35, 42]
        
    return sorted(list(best_indices))

# EVOLVE-BLOCK-END
