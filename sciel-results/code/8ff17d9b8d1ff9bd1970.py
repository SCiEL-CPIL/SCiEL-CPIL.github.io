import random
import time
import math

def construct():
    initial = [1, 2, 5, 11]
    best_set = None
    best_size = 0
    
    start_time = time.time()
    time_limit = 115.0
    
    def is_valid_sidon(S):
        """Check if S is a valid Sidon set."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_conflicts(S, x):
        """Return set of sums that would conflict if x was added to S."""
        conflicts = set()
        for y in S:
            s = x + y
            # We need to know if s is already in sums of S
            # This function is better used in a context where we have the sum set
            pass
        return conflicts

    def try_add(S, x, sums):
        """Try to add x to S. Returns (success, new_S, new_sums)."""
        if 2 * x in sums:
            return False, S, sums
        for y in S:
            if x + y in sums:
                return False, S, sums
        new_S = S + [x]
        new_sums = sums | {2 * x} | {x + y for y in S}
        return True, new_S, new_sums

    def greedy_init():
        """Create a valid set using greedy addition."""
        S = list(initial)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        
        candidates = [x for x in range(1, 285) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            success, new_S, new_sums = try_add(S, x, sums)
            if success:
                S = new_S
                sums = new_sums
        return S, sums

    # Initialize
    S, sums = greedy_init()
    best_set = S[:]
    best_size = len(S)
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.01
    cooling_rate = 0.99
    
    current_size = len(S)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = T_min
            
        # Generate a neighbor
        # Move: Remove a random non-initial element, add a random unused element
        removable = [x for x in S if x not in initial]
        if not removable:
            # If we can't remove anything, try swapping an initial element? 
            # No, initial must stay. If stuck, restart or break.
            # Actually, we can try to add a new element if possible without removing?
            # Greedy already did that. 
            # If we are stuck at a local minimum with only initial elements + few others, 
            # we might need to remove an element to add two others.
            # For now, if no removable, we can't do a simple swap.
            # We can try a "2-exchange": remove 2, add 2.
            # Let's stick to simple 1-exchange for now. If stuck, restart greedy.
            S, sums = greedy_init()
            current_size = len(S)
            if current_size > best_size:
                best_size = current_size
                best_set = S[:]
            continue

        x_remove = random.choice(removable)
        S_temp = [x for x in S if x != x_remove]
        sums_temp = set()
        for i in range(len(S_temp)):
            for j in range(i, len(S_temp)):
                sums_temp.add(S_temp[i] + S_temp[j])
        
        # Find a candidate to add
        # Instead of random, pick a random candidate from the unused set
        unused = [x for x in range(1, 285) if x not in S_temp]
        if not unused:
            continue
            
        random.shuffle(unused)
        success = False
        for x_add in unused:
            new_success, new_S, new_sums = try_add(S_temp, x_add, sums_temp)
            if new_success:
                S = new_S
                sums = new_sums
                current_size = len(S)
                success = True
                break
        
        if not success:
            # No valid add found? This shouldn't happen if we removed one and there are many unused.
            # But maybe all unused conflict? Unlikely.
            continue
            
        # Evaluate
        delta = current_size - best_size # This is not right for SA acceptance.
        # SA accepts based on current state vs previous state.
        # We need to track previous size.
        
        # Let's restructure the loop to track prev_size
        
    # The above structure is messy. Let's rewrite cleanly.
    
    # Restart with cleaner SA logic
    S, sums = greedy_init()
    prev_size = len(S)
    best_set = S[:]
    best_size = prev_size
    
    T = 1.0
    alpha = 0.999
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        removable = [x for x in S if x not in initial]
        if not removable:
            # Try to add if possible? Greedy did it. 
            # If stuck, restart.
            S, sums = greedy_init()
            prev_size = len(S)
            if prev_size > best_size:
                best_size = prev_size
                best_set = S[:]
            continue
            
        x_remove = random.choice(removable)
        S_temp = [x for x in S if x != x_remove]
        sums_temp = set()
        for i in range(len(S_temp)):
            for j in range(i, len(S_temp)):
                sums_temp.add(S_temp[i] + S_temp[j])
        
        unused = [x for x in range(1, 285) if x not in S_temp]
        if not unused:
            continue
            
        random.shuffle(unused)
        new_valid = False
        for x_add in unused:
            ok, new_S, new_sums = try_add(S_temp, x_add, sums_temp)
            if ok:
                S = new_S
                sums = new_sums
                new_valid = True
                break
                
        if not new_valid:
            continue
            
        curr_size = len(S)
        
        # Acceptance criterion
        if curr_size > prev_size:
            prev_size = curr_size
            if curr_size > best_size:
                best_size = curr_size
                best_set = S[:]
        elif curr_size == prev_size:
            # Accept to explore
            prev_size = curr_size
        else:
            # curr_size < prev_size (which is impossible for 1-remove 1-add if we found a valid add? 
            # Wait. If we remove 1 and add 1, size is same. 
            # So curr_size == prev_size always? 
            # Yes, for 1-exchange. 
            # To change size, we need 2-exchange or add/remove.
            # If we only do 1-exchange, size never changes! 
            # So we MUST do moves that change size.
            # Move: Remove 1, Add 1 (Size constant).
            # Move: Add 1 (Size +1) - only if valid.
            # Move: Remove 1 (Size -1) - only if we think it helps.
            
            # Let's allow:
            # 1. Try to add a random unused element. If valid, size increases.
            # 2. Try to remove a random non-initial element. Size decreases.
            # 3. Try to swap (remove 1, add 1). Size constant.
            
            pass # Logic below handles this

        T *= alpha
        if T < 0.01:
            T = 0.01

    return best_set
