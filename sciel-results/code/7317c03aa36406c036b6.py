import random
import time

N = 256

def is_valid_addition(x, S, sums):
    """
    Check if x can be added to S while maintaining Sidon property.
    S is a set of integers.
    sums is a set of all pairwise sums a+b for a,b in S, a<=b.
    Returns True if valid, False otherwise.
    """
    # New sums involving x: x+a for a in S, and 2*x
    new_sums = [x + a for a in S] + [2 * x]
    
    # Check for internal conflicts in new_sums
    if len(set(new_sums)) != len(new_sums):
        return False
        
    # Check for conflicts with existing sums
    for s in new_sums:
        if s in sums:
            return False
            
    return True

def maximize(S, sums):
    """
    Given a Sidon set S and its sums, add any missing element from [1, N]
    that preserves the Sidon property, until no more can be added.
    Returns the maximal Sidon set and its sums.
    """
    S = set(S)
    sums = set(sums)
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x in S:
                continue
            if is_valid_addition(x, S, sums):
                # Add x
                S.add(x)
                for a in S:
                    if a == x:
                        sums.add(2 * x)
                    elif a < x:
                        sums.add(x + a)
                    # Note: if a > x, x+a is already added when a was processed? 
                    # No, we are adding x now. We need to add x+a for all a in S (excluding x itself? No, a<=x).
                    # Let's just recalculate or carefully update.
                    # Efficient update:
                    # For each a in S:
                    #   if a < x: sums.add(a+x)
                    #   if a == x: sums.add(2*x)
                    #   if a > x: sums.add(a+x) is NOT needed yet? 
                    #   Wait, the set of sums is symmetric. 
                    #   If we add x, we must add x+a for all a in S.
                    #   Since we are in a loop, let's just do it correctly.
                    
                # Let's restart the addition logic to be safe and simple, N is small.
                # Actually, let's just compute the new sums properly.
                new_sums = set()
                for a in S:
                    new_sums.add(a + x)
                new_sums.add(2 * x)
                sums.update(new_sums)
                
                changed = True
                # Break and restart loop to check other elements again?
                # Or continue? If we add x, it might block y, but allow z?
                # Greedy addition order doesn't matter for final maximality?
                # Actually, it might. But any maximal extension is valid.
                # We just need *a* maximal set.
                break # Restart to ensure we don't miss interactions? 
    return S, sums

def construct():
    start_time = time.time()
    time_limit = 55.0 # Safety margin for 60s limit
    
    best_set = None
    best_size = N + 1
    
    # Helper to compute sums for a set
    def compute_sums(S):
        S = list(S)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    # Initial deterministic greedy to have a baseline
    S0 = set()
    sums0 = set()
    for x in range(1, N + 1):
        if is_valid_addition(x, S0, sums0):
            S0.add(x)
            # Update sums
            for a in S0:
                if a <= x:
                    sums0.add(a + x)
            sums0.add(2*x) # Wait, a=x is in S0, so x+x is covered if a=x.
            # Let's just use compute_sums for safety in the helper, or fix the update.
            # The update above: for a in S0 (which now includes x), if a<=x, add a+x.
            # If a=x, add 2x. If a<x, add a+x.
            # This is correct.
            
    S0, sums0 = maximize(S0, sums0)
    if len(S0) < best_size:
        best_size = len(S0)
        best_set = S0

    # Randomized search
    for _ in range(10000): # Max iterations, controlled by time
        if time.time() - start_time > time_limit:
            break
            
        # 1. Generate random Sidon set
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        S_rand = set()
        sums_rand = set()
        
        for x in elements:
            if is_valid_addition(x, S_rand, sums_rand):
                S_rand.add(x)
                for a in S_rand:
                    if a <= x:
                        sums_rand.add(a + x)
                sums_rand.add(2*x)
                
        # 2. Maximize it
        S_max, sums_max = maximize(S_rand, sums_rand)
        
        current_size = len(S_max)
        if current_size < best_size:
            best_size = current_size
            best_set = S_max
            
        # 3. Local Search (Swapping)
        # Try to improve S_max
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            S_curr = S_max
            sums_curr = sums_max
            curr_size = len(S_curr)
            
            # Try swapping each element in S_curr with each missing element
            missing = [x for x in range(1, N + 1) if x not in S_curr]
            
            # To speed up, we can iterate in random order
            elements_to_swap = list(S_curr)
            random.shuffle(elements_to_swap)
            
            for a in elements_to_swap:
                if time.time() - start_time > time_limit:
                    break
                random.shuffle(missing)
                for b in missing:
                    if time.time() - start_time > time_limit:
                        break
                    # Try S' = S_curr - {a} + {b}
                    if b in S_curr: continue # b is missing, so this is false
                    
                    # Check if S' is Sidon
                    # Remove a from sums
                    # This is complex to do incrementally. 
                    # Given N=256, recomputing sums for S' is O(k^2) where k~20.
                    # 400 ops. Very fast.
                    
                    S_test = S_curr - {a}
                    S_test.add(b)
                    
                    # Check Sidon property of S_test
                    valid = True
                    sums_test = set()
                    S_list = list(S_test)
                    for i in range(len(S_list)):
                        for j in range(i, len(S_list)):
                            s = S_list[i] + S_list[j]
                            if s in sums_test:
                                valid = False
                                break
                            sums_test.add(s)
                        if not valid:
                            break
                    
                    if valid:
                        # Maximize S_test
                        S_new, sums_new = maximize(S_test, sums_test)
                        if len(S_new) < curr_size:
                            S_max = S_new
                            sums_max = sums_new
                            curr_size = len(S_new)
                            improved = True
                            break # Restart swap loop with new S_max
                if improved:
                    break
            if time.time() - start_time > time_limit:
                break
        
        if len(S_max) < best_size:
            best_size = len(S_max)
            best_set = S_max

    return list(best_set)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 55.0 # Safety margin for 60s limit
    
    best_set = None
    best_size = N + 1
    
    def compute_sums(S):
        S = list(S)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def is_valid_addition(x, S, sums):
        new_sums = [x + a for a in S] + [2 * x]
        if len(set(new_sums)) != len(new_sums):
            return False
        for s in new_sums:
            if s in sums:
                return False
        return True

    def maximize(S, sums):
        S = set(S)
        sums = set(sums)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x in S:
                    continue
                if is_valid_addition(x, S, sums):
                    S.add(x)
                    # Update sums
                    for a in S:
                        if a <= x:
                            sums.add(a + x)
                    sums.add(2*x)
                    changed = True
                    break
        return S, sums

    # Initial deterministic greedy
    S0 = set()
    sums0 = set()
    for x in range(1, N + 1):
        if is_valid_addition(x, S0, sums0):
            S0.add(x)
            for a in S0:
                if a <= x:
                    sums0.add(a + x)
            sums0.add(2*x)
            
    S0, sums0 = maximize(S0, sums0)
    if len(S0) < best_size:
        best_size = len(S0)
        best_set = S0

    # Randomized search
    for _ in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        S_rand = set()
        sums_rand = set()
        
        for x in elements:
            if is_valid_addition(x, S_rand, sums_rand):
                S_rand.add(x)
                for a in S_rand:
                    if a <= x:
                        sums_rand.add(a + x)
                sums_rand.add(2*x)
                
        S_max, sums_max = maximize(S_rand, sums_rand)
        
        current_size = len(S_max)
        if current_size < best_size:
            best_size = current_size
            best_set = S_max
            
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            S_curr = S_max
            sums_curr = sums_max
            curr_size = len(S_curr)
            
            missing = [x for x in range(1, N + 1) if x not in S_curr]
            elements_to_swap = list(S_curr)
            random.shuffle(elements_to_swap)
            
            for a in elements_to_swap:
                if time.time() - start_time > time_limit:
                    break
                random.shuffle(missing)
                for b in missing:
                    if time.time() - start_time > time_limit:
                        break
                    
                    S_test = S_curr - {a}
                    S_test.add(b)
                    
                    valid = True
                    sums_test = set()
                    S_list = list(S_test)
                    for i in range(len(S_list)):
                        for j in range(i, len(S_list)):
                            s = S_list[i] + S_list[j]
                            if s in sums_test:
                                valid = False
                                break
                            sums_test.add(s)
                        if not valid:
                            break
                    
                    if valid:
                        S_new, sums_new = maximize(S_test, sums_test)
                        if len(S_new) < curr_size:
                            S_max = S_new
                            sums_max = sums_new
                            curr_size = len(S_new)
                            improved = True
                            break
                if improved:
                    break
            if time.time() - start_time > time_limit:
                break
        
        if len(S_max) < best_size:
            best_size = len(S_max)
            best_set = S_max

    return list(best_set)
# EVOLVE-BLOCK-END
