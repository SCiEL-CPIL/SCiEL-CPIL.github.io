import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    required = [1, 2, 5, 11]
    max_val = 284
    all_candidates = set(range(1, max_val + 1))
    
    # Precompute required sums
    req_sums = set()
    for i in range(len(required)):
        for j in range(i, len(required)):
            req_sums.add(required[i] + required[j])
            
    init_mask = 0
    for s in req_sums:
        init_mask |= (1 << s)
        
    best_set = required[:]
    best_size = len(required)
    
    def is_valid_addition(x, curr_set, mask):
        # Check if x can be added to curr_set
        if x in curr_set:
            return False
        # Check 2x
        if mask & (1 << (2 * x)):
            return False
        # Check x + y
        for y in curr_set:
            if mask & (1 << (x + y)):
                return False
        return True

    def try_greedy_extension(curr_set, mask, candidates):
        """Try to add as many candidates as possible in the given order."""
        for c in candidates:
            if time.time() > deadline:
                return curr_set, mask
            if is_valid_addition(c, curr_set, mask):
                curr_set.append(c)
                mask |= (1 << (2 * c))
                for y in curr_set[:-1]:
                    mask |= (1 << (c + y))
        return curr_set, mask

    # Strategy: Multiple random orderings + Local Search
    # We will generate random permutations of the non-required candidates
    # and run a greedy pass, followed by a local search phase.
    
    base_candidates = [x for x in range(1, max_val + 1) if x not in required]
    
    # Run several iterations
    num_iterations = 500
    for i in range(num_iterations):
        if time.time() > deadline:
            break
            
        # Randomize the order of candidates
        random.shuffle(base_candidates)
        
        curr_set = required[:]
        curr_mask = init_mask
        
        # 1. Greedy Phase
        curr_set, curr_mask = try_greedy_extension(curr_set, curr_mask, base_candidates)
        
        if len(curr_set) > best_size:
            best_set = curr_set[:]
            best_size = len(curr_set)
            
        # 2. Local Search Phase: Try to improve by swapping
        # We try to find a swap x -> y that allows adding a new z.
        # To keep it fast, we limit the depth of the search or the number of swaps.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            curr_set_copy = curr_set[:]
            
            # Try adding any remaining candidate first
            for c in base_candidates:
                if time.time() > deadline:
                    break
                if is_valid_addition(c, curr_set, curr_mask):
                    curr_set.append(c)
                    curr_mask |= (1 << (2 * c))
                    for y in curr_set[:-1]:
                        curr_mask |= (1 << (c + y))
                    improved = True
                    if len(curr_set) > best_size:
                        best_set = curr_set[:]
                        best_size = len(curr_set)
                    break # Restart the loop to try adding more
            
            if improved:
                continue
                
            # If no direct addition, try swaps
            # To avoid O(N^3) complexity, we sample swaps or limit attempts
            # Let's try swapping each x in curr_set with a random y not in curr_set
            # If the swap is valid, check if it allows an addition.
            
            # Shuffle x for randomization
            x_list = curr_set[:]
            random.shuffle(x_list)
            
            for x in x_list:
                if time.time() > deadline:
                    break
                
                # Remove x temporarily
                # Reconstruct mask without x is expensive. 
                # Instead, we can just check validity of y against (curr_set \ {x})
                # We need a fast way to check if y is valid against curr_set \ {x}.
                
                # Let's just try random y's for this x
                # Sample a few y's
                for _ in range(10): # Sample 10 random y's
                    if time.time() > deadline:
                        break
                    y = random.choice(base_candidates)
                    if y in curr_set:
                        continue
                    if y == x:
                        continue
                    
                    # Check if y is valid against curr_set \ {x}
                    # This requires checking y against all z in curr_set, z != x
                    valid = True
                    # Check 2y
                    # We need to know if 2y is in sums of curr_set \ {x}
                    # This is tricky with the global mask.
                    # Alternative: Just construct the new set and check validity from scratch?
                    # Size is ~20. O(K^2) is 400 ops. Very fast.
                    
                    new_set = [z for z in curr_set if z != x]
                    new_set.append(y)
                    
                    # Check if new_set is a valid Sidon set
                    sums = set()
                    valid = True
                    for i in range(len(new_set)):
                        a = new_set[i]
                        for j in range(i, len(new_set)):
                            b = new_set[j]
                            s = a + b
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # If valid, check if we can add any z to new_set
                    # If yes, accept the swap and the addition
                    added_z = None
                    for z in base_candidates:
                        if z in new_set:
                            continue
                        # Check if z can be added to new_set
                        # Re-check sums
                        # Optimization: We already have sums for new_set.
                        # Check 2z and z+u for u in new_set
                        if 2*z in sums:
                            continue
                        conflict = False
                        for u in new_set:
                            if (z+u) in sums:
                                conflict = True
                                break
                        if not conflict:
                            added_z = z
                            break
                    
                    if added_z is not None:
                        # Accept swap x -> y and add z
                        curr_set = new_set[:]
                        curr_set.append(added_z)
                        
                        # Update mask for curr_set
                        curr_mask = 0
                        for a in curr_set:
                            for b in curr_set:
                                if a <= b:
                                    curr_mask |= (1 << (a+b))
                        
                        if len(curr_set) > best_size:
                            best_set = curr_set[:]
                            best_size = len(curr_set)
                        
                        improved = True
                        break # Break out of y loop, restart while loop
                    # If no z added, we might still accept the swap if we are doing SA, 
                    # but for simplicity and speed, we only accept if it leads to growth.
                
                if improved:
                    break
        
    return sorted(best_set)
