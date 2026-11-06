import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # Prescribed elements
    base = [1, 2, 5, 11]
    base_set = set(base)
    
    # Helper to check validity and get sums
    def get_validity_and_sums(candidates):
        n = len(candidates)
        sums = set()
        for i in range(n):
            a = candidates[i]
            for j in range(i, n):
                b = candidates[j]
                s = a + b
                if s in sums:
                    return False, None
                sums.add(s)
        return True, sums

    # Initial valid set is just the base
    current_set = set(base)
    best_size = len(base)
    best_solution = base.copy()
    
    # We want to maximize the size.
    # Max possible size in [1, 284] is 20.
    # With constraints 1, 2, 5, 11, it's likely around 18-20.
    
    # Strategy:
    # 1. Try to add elements one by one using a randomized backtracking/heuristic.
    # 2. Use Simulated Annealing to rearrange the set to allow more additions.
    
    # Since the base is fixed, we can try to find a large Golomb ruler containing these 4 points.
    
    # A known approach for Golomb rulers is to start with a known optimal ruler
    # and try to fit the constraints, or start with constraints and add marks.
    
    # Let's try a randomized local search starting from a valid set of size 4.
    # We will attempt to grow the set. If stuck, we'll try to swap out non-base elements
    # to create space for more elements.
    
    def random_valid_extension(current, target_size):
        """
        Attempts to extend 'current' (a set of ints) to size 'target_size'
        by adding random valid integers from [1, 284].
        Returns the new set if successful, else None.
        """
        curr = list(current)
        curr.sort()
        valid, sums = get_validity_and_sums(curr)
        if not valid:
            return None
        
        candidates = [i for i in range(1, 285) if i not in current]
        random.shuffle(candidates)
        
        attempts = 0
        max_attempts = 5000 # Limit per restart attempt to avoid wasting time
        
        while len(curr) < target_size and attempts < max_attempts:
            attempts += 1
            # Pick a random candidate
            # To improve efficiency, we can prioritize candidates that don't conflict immediately
            # But simple random is okay for a start.
            
            # Actually, picking from the full list is slow. 
            # Let's pick a random index from the remaining candidates list.
            if not candidates:
                break
                
            # We need to be careful: candidates list shrinks as we add elements.
            # Let's just pick a random element from the remaining pool.
            idx = random.randint(0, len(candidates) - 1)
            x = candidates[idx]
            
            # Check if x can be added
            # New sums: x + y for all y in curr, and 2*x
            new_sums = {x + y for y in curr} | {2 * x}
            
            # Check for conflicts
            # 1. Internal conflicts in new_sums (all distinct by construction if x != y, but 2x is unique)
            # Actually {x+y for y in curr} are distinct if curr elements are distinct.
            # 2. Conflict with existing sums
            if not (new_sums & sums):
                # Valid addition
                curr.append(x)
                curr.sort()
                sums |= new_sums
                candidates.pop(idx)
            else:
                # Invalid, remove from candidates to avoid re-picking immediately?
                # Or just keep it? Keeping it might lead to many failed attempts.
                # Let's remove it to force exploration of other numbers.
                candidates.pop(idx)
                
        if len(curr) == target_size:
            return set(curr)
        return None

    # We want to maximize size. Let's try to reach size 20, then 19, etc.
    # Since we have 4 fixed points, we need to add up to 16 more.
    
    # It is often better to start with a larger valid set if possible.
    # But we must contain 1, 2, 5, 11.
    
    # Let's try a different strategy: 
    # Generate random Golomb rulers of size K, and check if they contain 1,2,5,11.
    # This is unlikely to hit exactly.
    
    # Better: Start with {1, 2, 5, 11} and try to expand.
    # If we get stuck at size S, we can try to "swap" a non-base element with a different one
    # to see if that allows further expansion.
    
    # Let's implement a simple hill-climbing with swaps.
    
    def try_expand(current_set):
        """
        Greedily try to add elements to current_set.
        Returns the expanded set.
        """
        curr = list(current_set)
        curr.sort()
        valid, sums = get_validity_and_sums(curr)
        if not valid:
            return current_set
        
        remaining = [i for i in range(1, 285) if i not in current_set]
        random.shuffle(remaining)
        
        # Try to add as many as possible
        # We iterate through remaining, and if it fits, add it.
        # This is greedy and order-dependent.
        # To make it stochastic, we shuffle remaining.
        
        added = 0
        for x in remaining:
            if len(curr) >= 20: # Max possible
                break
            new_sums = {x + y for y in curr} | {2 * x}
            if not (new_sums & sums):
                curr.append(x)
                curr.sort()
                sums |= new_sums
                added += 1
        
        return set(curr)

    # Main loop: Random restarts with expansion
    # Since the base is fixed, we don't restart the base, but we restart the *order* of addition
    # or use a more complex search.
    
    # Actually, the greedy expansion above is deterministic in result if the shuffle is fixed.
    # We need to explore different "paths".
    
    # Let's try to find a solution of size 19 or 20.
    # We'll run multiple attempts of "randomized greedy expansion".
    
    best_size_found = len(base)
    best_sol_found = base.copy()
    
    # Number of restarts
    # Each restart: shuffle candidates, try to expand.
    # If we get stuck, we might try to "kick" the solution by removing a random non-base element
    # and trying to re-expand.
    
    def kick_and_expand(current_set):
        """
        Remove a random non-base element, then try to re-expand.
        """
        if len(current_set) <= len(base):
            return current_set
            
        non_base = [x for x in current_set if x not in base_set]
        if not non_base:
            return current_set
            
        x_to_remove = random.choice(non_base)
        new_set = current_set - {x_to_remove}
        
        # Try to expand the new set
        expanded = try_expand(new_set)
        return expanded

    # Simulated annealing with swap operations
    def simulated_annealing(start_set, max_iterations=5000):
        """
        Use simulated annealing to find a larger Sidon set by swapping non-base elements.
        """
        if time.time() > deadline - 2:
            return start_set
            
        curr = set(start_set)
        curr_list = sorted(curr)
        valid, curr_sums = get_validity_and_sums(curr_list)
        if not valid:
            return start_set
        
        best = set(curr)
        best_size = len(curr)
        best_sums = set(curr_sums)
        
        temperature = 100.0
        cooling = 0.95
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2:
                break
                
            if len(curr) <= len(base):
                break
                
            non_base = [x for x in curr if x not in base_set]
            if not non_base:
                break
                
            # Pick a random non-base element to potentially swap
            x_to_swap = random.choice(non_base)
            
            # Pick a random candidate to swap with
            candidates = [i for i in range(1, 285) if i not in curr]
            if not candidates:
                break
            new_x = random.choice(candidates)
            
            # Try the swap
            new_set = (curr - {x_to_swap}) | {new_x}
            new_list = sorted(new_set)
            new_valid, new_sums = get_validity_and_sums(new_list)
            
            if new_valid:
                # Check if we can add more elements after the swap
                expanded = try_expand(new_set)
                if len(expanded) > best_size:
                    best = expanded
                    best_size = len(expanded)
                    best_sums = set()
                    # Recompute sums for best
                    _, best_sums = get_validity_and_sums(sorted(best))
                    curr = expanded
                    curr_list = sorted(curr)
                    valid, curr_sums = get_validity_and_sums(curr_list)
                elif len(expanded) == len(curr):
                    # Accept with probability based on temperature
                    if random.random() < temperature:
                        curr = new_set
                        curr_list = sorted(curr)
                        valid, curr_sums = get_validity_and_sums(curr_list)
            else:
                # Invalid swap, accept with small probability
                if random.random() < temperature * 0.1:
                    # This shouldn't happen since we check validity
                    pass
            
            temperature *= cooling
            
            if len(curr) > best_size:
                best = set(curr)
                best_size = len(curr)
                _, best_sums = get_validity_and_sums(sorted(best))
        
        return best

    # Let's run a loop
    # We'll maintain a set of "promising" solutions (beam width 1 for simplicity, but maybe 5)
    # Actually, just running many independent randomized greedy expansions is often effective.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Strategy 1: Pure randomized greedy expansion from base
        # Shuffle the candidate pool implicitly by shuffling the range
        # We can simulate this by shuffling the list of candidates in try_expand
        
        # To add more diversity, let's occasionally use a "kick" strategy
        # Start with base, expand. If size < best_size, maybe try to kick?
        # But kicking from base does nothing (no non-base elements).
        
        # So we need to get to a larger size first.
        # Let's just do randomized greedy expansion.
        
        sol = try_expand(base_set)
        
        if len(sol) > best_size_found:
            best_size_found = len(sol)
            best_sol_found = sol.copy()
            # If we reached 20, we can stop? 
            # The max is 20. If we get 20, it's optimal.
            if best_size_found == 20:
                return sorted(best_sol_found)
                
        # If we are stuck at a size less than best, we can try to improve the current best
        # by kicking.
        # But only if we have a current best that is larger than base.
        if best_size_found > len(base):
            # Try to kick the best solution found so far
            # This might find a different arrangement of the same size or larger
            kicked = kick_and_expand(set(best_sol_found))
            if len(kicked) > best_size_found:
                best_size_found = len(kicked)
                best_sol_found = kicked.copy()
                if best_size_found == 20:
                    return sorted(best_sol_found)
            
            # Also, try simulated annealing on the best solution
            if time.time() < deadline - 10:
                sa_result = simulated_annealing(set(best_sol_found))
                if len(sa_result) > best_size_found:
                    best_size_found = len(sa_result)
                    best_sol_found = sa_result.copy()
                    if best_size_found == 20:
                        return sorted(best_sol_found)
            
            # Also, try to kick a *random* valid extension of base?
            # No, just kicking the best is a form of local search.
            
        # If time is running out, break
        if time.time() > deadline - 5:
            break

    return sorted(best_sol_found)

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We need to ensure the function name matches.
# The prompt asks to keep function names and signatures.
# The seed program had `def construct():`
# My code has `def construct():`
# I should wrap it in the block if required, but the instruction says "Keep the function names... of the seed program".
# The seed program had the function inside the block.
# I will place the code inside the block.
# EVOLVE-BLOCK-END
