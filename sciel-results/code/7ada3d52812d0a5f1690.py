import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 1000
    max_sum = 2 * N  # 2000
    # counts[s] = number of pairs (a,b) in S with a+b=s
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    # Initial greedy construction (ascending order often yields good density)
    S = []
    S_set = set()
    
    # We need a fast way to check if x can be added.
    # Condition: for all y in S, counts[x+y] < 2 AND counts[2x] < 2.
    # Since S is small initially, we can just loop.
    # As S grows, we might need optimization, but |S| is ~500, so 500 checks is fine.
    
    # To speed up initial greedy, we can pre-compute something? 
    # No, just iterate.
    
    # Let's use a list for S and a set for O(1) membership.
    # Actually, for the vectorized add check:
    # If we have S as a numpy array, we can do:
    # sums = x + S_array
    # if np.all(counts[sums] < 2) and counts[2*x] < 2: add
    
    S_array = np.array([], dtype=np.int32)
    
    # Greedy Phase 1: Ascending
    for x in range(1, N + 1):
        if len(S_array) == 0:
            S_array = np.array([x], dtype=np.int32)
            S_set.add(x)
            counts[2*x] += 1
            continue
        
        # Check 2x
        if counts[2*x] >= 2:
            continue
        
        # Check x + y for all y in S
        # S_array contains elements. Note: S_array is not necessarily sorted, but greedy ascending keeps it sorted.
        # sums = x + S_array
        # We need to check counts[sums] < 2 for all sums.
        # Also, the pair (x,x) is handled by counts[2x].
        # The pairs (x,y) for y in S are sums x+y.
        # If y == x, sum is 2x. We already checked counts[2x] < 2.
        # Wait, if y == x is in S, then counts[2x] includes the pair (x,x).
        # When we add x, we add pairs (x,y) for all y in S.
        # If x is already in S? No, we are adding distinct integers.
        # So y != x for all y in S.
        # Thus sums = x + S_array does not include 2x.
        # So we check:
        sums = x + S_array
        if np.all(counts[sums] < 2):
            # Safe to add
            S_array = np.append(S_array, x)
            S_set.add(x)
            counts[2*x] += 1
            counts[sums] += 1
    
    best_size = len(S_array)
    best_set = set(S_array)
    
    # Simulated Annealing Phase
    # Objective: Maximize size.
    # Moves:
    # 1. Add a random element from complement.
    # 2. Remove a random element from S.
    # 3. Swap: Remove x, Add y.
    
    # We need efficient add/remove.
    # Using a Python list for S is slow for numpy ops.
    # Using a numpy array is fast for ops but slow for insertion/deletion.
    # Given |S| ~ 500, we can convert to list for set ops and back?
    # Or just keep a Python list and convert to numpy array for the check?
    # Converting list of 500 ints to numpy array is fast.
    
    S_list = list(S_array)
    
    def try_add(x, S_list, counts):
        """Returns True if x can be added, and updates state if so."""
        if x in S_set:
            return False
        if counts[2*x] >= 2:
            return False
        if len(S_list) == 0:
            S_list.append(x)
            S_set.add(x)
            counts[2*x] += 1
            return True
        
        S_np = np.array(S_list, dtype=np.int32)
        sums = x + S_np
        if np.any(counts[sums] >= 2):
            return False
        
        S_list.append(x)
        S_set.add(x)
        counts[2*x] += 1
        counts[sums] += 1
        return True

    def try_remove(x, S_list, counts):
        """Removes x from S if present."""
        if x not in S_set:
            return False
        S_list.remove(x)
        S_set.remove(x)
        counts[2*x] -= 1
        if len(S_list) > 0:
            S_np = np.array(S_list, dtype=np.int32)
            sums = x + S_np
            counts[sums] -= 1
        return True

    T = 10.0
    alpha = 0.995
    
    while time.time() < deadline:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add
            x = random.randint(1, N)
            if x not in S_set:
                # Check validity
                if counts[2*x] < 2:
                    if len(S_list) == 0:
                        valid = True
                    else:
                        S_np = np.array(S_list, dtype=np.int32)
                        sums = x + S_np
                        valid = np.all(counts[sums] < 2)
                    
                    if valid:
                        # Perform add
                        S_list.append(x)
                        S_set.add(x)
                        counts[2*x] += 1
                        if len(S_list) > 1:
                            # Recalculate sums for counts update? 
                            # We already calculated sums for check. 
                            # But S_list changed? No, we check before adding.
                            # So sums are correct for the state BEFORE adding.
                            # We need to update counts for the NEW state.
                            # The new pairs are (x, y) for y in OLD S.
                            # So counts[sums] += 1 is correct.
                            S_np_old = np.array([y for y in S_list if y != x], dtype=np.int32)
                            sums_update = x + S_np_old
                            counts[sums_update] += 1
                        else:
                            # Only x in S
                            pass
                        
                        # Update best
                        if len(S_list) > best_size:
                            best_size = len(S_list)
                            best_set = set(S_list)
                        
                    else:
                        # Invalid, do nothing (or accept with prob exp(-1/T)? No, it's infeasible)
                        pass
                else:
                    pass

        elif move_type < 0.7:
            # Try to remove
            if S_list:
                x = random.choice(S_list)
                # Remove
                S_list.remove(x)
                S_set.remove(x)
                counts[2*x] -= 1
                if S_list:
                    S_np = np.array(S_list, dtype=np.int32)
                    sums = x + S_np
                    counts[sums] -= 1
                
                # Should we keep it removed? 
                # If we remove, size decreases. We only accept if we can add something better later?
                # In SA, we accept worse states with prob exp(-delta/T).
                # Here delta = -1 (size decreases by 1).
                # Prob = exp(-(-1)/T) = exp(1/T) > 1. So we always accept removal?
                # That's not right. Removal makes the set smaller, which is worse.
                # We should accept removal with probability exp(-1/T) if we treat "cost" as -size.
                # Let's define energy E = -size. Lower E is better.
                # Delta E = -1 (size -1).
                # Accept if rand() < exp(-Delta E / T) = exp(1/T). This is > 1.
                # Wait. If we remove, E becomes - (size-1) = -size + 1.
                # Old E = -size.
                # Delta E = E_new - E_old = 1.
                # Accept if rand() < exp(-1/T).
                
                if random.random() > np.exp(-1.0 / T):
                    # Reject removal, put it back
                    # This is inefficient. 
                    # Better: Only propose removal if it allows a subsequent add?
                    # Or just propose swaps.
                    
                    # Let's revert
                    S_list.append(x)
                    S_set.add(x)
                    counts[2*x] += 1
                    if S_list:
                         # Note: S_list now contains x.
                         # We need to add back counts for x+y for y in S_list \ {x}
                         S_np = np.array([y for y in S_list if y != x], dtype=np.int32)
                         sums = x + S_np
                         counts[sums] += 1
                else:
                    # Accept removal
                    pass

        else:
            # Swap: Remove x, Add y
            if S_list:
                x = random.choice(S_list)
                y = random.randint(1, N)
                if y != x and y not in S_set:
                    # Check if y can be added to S \ {x}
                    S_temp = [z for z in S_list if z != x]
                    
                    valid = True
                    if counts[2*x] >= 2:
                        # We are removing x, so counts[2x] will decrease by 1.
                        # If counts[2x] was 2, it becomes 1. So 2y check?
                        # We need to check if y can be added to S_temp.
                        # Condition: counts_temp[2y] < 2 and counts_temp[y+z] < 2 for all z in S_temp.
                        # counts_temp[s] = counts[s] - (1 if s involves x else 0)
                        
                        # Let's compute the change in counts due to removing x.
                        # Removed pairs: (x,x) and (x,z) for z in S_temp.
                        
                        # Check 2y
                        c_2y = counts[2*y]
                        # Does removing x affect 2y?
                        # Only if 2y == 2x (y=x, impossible) or 2y == x+z.
                        # If x+z = 2y, then c_2y decreases by 1.
                        # This is complex. 
                        
                        # Easier: Just construct the hypothetical state and check.
                        # But that's expensive.
                        
                        # Alternative: Just try to remove x, then try to add y.
                        # If add fails, revert both.
                        
                        # 1. Remove x
                        S_list.remove(x)
                        S_set.remove(x)
                        counts[2*x] -= 1
                        if S_list:
                            S_np = np.array(S_list, dtype=np.int32)
                            sums_x = x + S_np
                            counts[sums_x] -= 1
                        
                        # 2. Try add y
                        added = False
                        if counts[2*y] < 2:
                            if len(S_list) == 0:
                                added = True
                            else:
                                S_np = np.array(S_list, dtype=np.int32)
                                sums_y = y + S_np
                                if np.all(counts[sums_y] < 2):
                                    added = True
                            
                            if added:
                                S_list.append(y)
                                S_set.add(y)
                                counts[2*y] += 1
                                if S_list:
                                    S_np = np.array([z for z in S_list if z != y], dtype=np.int32)
                                    sums_y = y + S_np
                                    counts[sums_y] += 1
                                
                                # Update best
                                if len(S_list) > best_size:
                                    best_size = len(S_list)
                                    best_set = set(S_list)
                        else:
                            # Add failed, revert remove
                            S_list.append(x)
                            S_set.add(x)
                            counts[2*x] += 1
                            if S_list:
                                S_np = np.array([z for z in S_list if z != x], dtype=np.int32)
                                sums_x = x + S_np
                                counts[sums_x] += 1
                    else:
                        # counts[2x] < 2, but we still need to check the rest.
                        # Same logic: Remove x, try add y.
                        S_list.remove(x)
                        S_set.remove(x)
                        counts[2*x] -= 1
                        if S_list:
                            S_np = np.array(S_list, dtype=np.int32)
                            sums_x = x + S_np
                            counts[sums_x] -= 1
                        
                        added = False
                        if counts[2*y] < 2:
                            if len(S_list) == 0:
                                added = True
                            else:
                                S_np = np.array(S_list, dtype=np.int32)
                                sums_y = y + S_np
                                if np.all(counts[sums_y] < 2):
                                    added = True
                            
                            if added:
                                S_list.append(y)
                                S_set.add(y)
                                counts[2*y] += 1
                                if S_list:
                                    S_np = np.array([z for z in S_list if z != y], dtype=np.int32)
                                    sums_y = y + S_np
                                    counts[sums_y] += 1
                                
                                if len(S_list) > best_size:
                                    best_size = len(S_list)
                                    best_set = set(S_list)
                        else:
                            S_list.append(x)
                            S_set.add(x)
                            counts[2*x] += 1
                            if S_list:
                                S_np = np.array([z for z in S_list if z != x], dtype=np.int32)
                                sums_x = x + S_np
                                counts[sums_x] += 1

        # Cool down
        T *= alpha
        if T < 0.1:
            T = 0.1

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))
