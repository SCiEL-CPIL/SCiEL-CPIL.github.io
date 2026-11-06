import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    base = [1, 2, 5, 11]
    base_set = set(base)
    
    def is_valid_set(s):
        """Check if set s is a valid Sidon set."""
        sums = set()
        for i in range(len(s)):
            si = s[i]
            for j in range(i, len(s)):
                sj = s[j]
                val = si + sj
                if val in sums:
                    return False
                sums.add(val)
        return True

    def is_valid_addition(x, current_list, current_sums):
        """Check if x can be added to current_list."""
        if 2 * x in current_sums:
            return False
        for y in current_list:
            if x + y in current_sums:
                return False
        return True

    def get_sums(lst):
        sums = set()
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums.add(lst[i] + lst[j])
        return sums

    best_solution = list(base)
    best_size = len(base)

    def greedy_fill(cands, current_list, current_sums, time_limit):
        """Greedily add candidates."""
        nonlocal best_solution, best_size
        
        i = 0
        while i < len(cands) and time.time() < time_limit:
            x = cands[i]
            if is_valid_addition(x, current_list, current_sums):
                current_list.append(x)
                current_sums.add(2 * x)
                for y in current_list[:-1]:
                    current_sums.add(x + y)
                
                if len(current_list) > best_size:
                    best_size = len(current_list)
                    best_solution = list(current_list)
                
            i += 1
        return current_list, current_sums

    def local_search(current_list, current_sums, time_limit):
        """
        Local search with 1-add, 1-swap, and 2-swap.
        """
        nonlocal best_solution, best_size
        
        current_list = list(current_list)
        current_sums = get_sums(current_list)
        
        # State variables
        current_set = set(current_list)
        movable_indices = [i for i, x in enumerate(current_list) if x not in base_set]
        candidates = [x for x in range(1, 285) if x not in current_set]
        
        attempts = 0
        max_attempts = 2000 # Limit iterations to prevent infinite loop in one LS
        
        while time.time() < time_limit and attempts < max_attempts:
            attempts += 1
            
            # 1. Try to increase size
            added = False
            # Randomly shuffle candidates to avoid bias? 
            # Or just iterate. Iterating is deterministic. 
            # Let's iterate.
            for idx, x in enumerate(candidates):
                if is_valid_addition(x, current_list, current_sums):
                    # Add x
                    current_list.append(x)
                    current_sums.add(2 * x)
                    for y in current_list[:-1]:
                        current_sums.add(x + y)
                    
                    if len(current_list) > best_size:
                        best_size = len(current_list)
                        best_solution = list(current_list)
                    
                    # Update state
                    current_set.add(x)
                    candidates.pop(idx)
                    movable_indices.append(len(current_list) - 1)
                    added = True
                    break
            
            if added:
                continue
            
            # 2. Try 1-swap
            if len(movable_indices) >= 1 and len(candidates) >= 1:
                r_idx = random.choice(movable_indices)
                r_val = current_list[r_idx]
                a_val = random.choice(candidates)
                
                # Check validity of swap
                # Remove r_val, add a_val
                # Since K is small, just check full set
                new_list = [x for i, x in enumerate(current_list) if i != r_idx]
                new_list.append(a_val)
                
                if is_valid_set(new_list):
                    # Accept
                    current_list = new_list
                    current_sums = get_sums(current_list)
                    current_set = set(current_list)
                    movable_indices = [i for i, x in enumerate(current_list) if x not in base_set]
                    candidates = [x for x in range(1, 285) if x not in current_set]
                    continue

            # 3. Try 2-swap
            if len(movable_indices) >= 2 and len(candidates) >= 2:
                r_idx1, r_idx2 = random.sample(movable_indices, 2)
                r1 = current_list[r_idx1]
                r2 = current_list[r_idx2]
                a1, a2 = random.sample(candidates, 2)
                
                new_list = [x for i, x in enumerate(current_list) if i != r_idx1 and i != r_idx2]
                new_list.append(a1)
                new_list.append(a2)
                
                if is_valid_set(new_list):
                    current_list = new_list
                    current_sums = get_sums(current_list)
                    current_set = set(current_list)
                    movable_indices = [i for i, x in enumerate(current_list) if x not in base_set]
                    candidates = [x for x in range(1, 285) if x not in current_set]
                    continue

            # If we reach here, no move was accepted in this attempt
            # We could break if stuck for too long, but the while loop handles time.
            # To avoid getting stuck in a specific bad state, we could restart LS?
            # No, the outer loop handles restarts.

        return current_list, current_sums

    # Main Loop
    restart_count = 0
    while time.time() < deadline:
        cands = [x for x in range(1, 285) if x not in base_set]
        random.shuffle(cands)
        
        current_list = list(base)
        current_sums = get_sums(current_list)
        
        # Greedy Phase
        current_list, current_sums = greedy_fill(cands, current_list, current_sums, deadline - 5)
        
        # Local Search Phase
        if time.time() < deadline - 2:
            current_list, current_sums = local_search(current_list, current_sums, deadline - 2)
        
        restart_count += 1
        
        if best_size >= 20:
            break

    return sorted(best_solution)

if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
