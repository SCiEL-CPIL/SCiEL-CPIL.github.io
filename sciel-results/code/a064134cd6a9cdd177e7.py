"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 40, 'maximum_size': 18}

def construct():
    """
    Construct a set of distinct integers in 2..40 with distinct subset products.
    Maximize the number of selected integers.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute exponent vectors for each number in 2..40
    vecs = {}
    for n in range(2, P['n'] + 1):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[n] = tuple(v)
        
    zero_vec = (0,) * num_primes
    
    # Helper to compute all subset sums for a given set of numbers
    def compute_sums(nums):
        sums = {zero_vec}
        for n in nums:
            v = vecs[n]
            new_sums = []
            for s in sums:
                # Add vector v to s
                ns = tuple(s[i] + v[i] for i in range(num_primes))
                new_sums.append(ns)
            sums.update(new_sums)
        return sums

    # Helper to check if adding a number keeps sums distinct
    def can_add(nums, sums, candidate):
        v = vecs[candidate]
        # Check if any new sum collides with existing sums
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            if ns in sums:
                return False, None
        # If no collision, generate the new sums
        new_sums = set()
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            new_sums.add(ns)
        return True, new_sums

    best_set = []
    best_size = 0
    
    # Initial heuristic: Start with primes (always valid)
    # Primes are unit vectors, so their subset sums are all distinct (each subset corresponds to a unique 0/1 vector in the prime dimensions)
    initial_set = primes[:]
    initial_sums = compute_sums(initial_set)
    best_set = initial_set
    best_size = len(initial_set)
    
    # Local Search Function
    def local_search(start_set, start_sums, time_budget):
        nonlocal best_set, best_size
        current_set = list(start_set)
        current_sums = set(start_sums) # Copy
        current_size = len(current_set)
        in_set = set(current_set)
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            if current_size >= P['maximum_size']:
                break
                
            improved = False
            
            # 1. Try to Add
            # Iterate through candidates not in set
            # Randomize order to avoid bias?
            candidates = [n for n in range(2, P['n'] + 1) if n not in in_set]
            random.shuffle(candidates)
            
            added = False
            for c in candidates:
                ok, new_sums = can_add(current_set, current_sums, c)
                if ok:
                    current_set.append(c)
                    current_sums.update(new_sums)
                    in_set.add(c)
                    current_size += 1
                    added = True
                    improved = True
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_set = list(current_set)
                    break # Restart the loop to try adding more
                    
            if added:
                continue
                
            # 2. Try to Swap (Remove one, Add one)
            # This helps escape local optima where no single addition works,
            # but replacing an element frees up "space" for two additions or a better configuration.
            # We iterate through elements to remove
            remove_candidates = list(current_set)
            random.shuffle(remove_candidates)
            
            swapped = False
            for r in remove_candidates:
                # Rebuild sums without r
                temp_set = [x for x in current_set if x != r]
                temp_sums = compute_sums(temp_set)
                
                # Try to add any candidate not in temp_set
                add_candidates = [n for n in range(2, P['n'] + 1) if n not in set(temp_set)]
                random.shuffle(add_candidates)
                
                for a in add_candidates:
                    ok, new_sums = can_add(temp_set, temp_sums, a)
                    if ok:
                        # Perform the swap
                        current_set = temp_set + [a]
                        current_sums = temp_sums
                        current_sums.update(new_sums)
                        in_set = set(current_set)
                        # Size remains the same, but we might be able to add more later
                        swapped = True
                        improved = True
                        break
                if swapped:
                    break
                    
            if swapped:
                continue
                
            # 3. Try to Remove (to allow future adds? Usually not directly helpful for max size 
            # unless we are stuck in a configuration where removing one allows adding 3, etc.)
            # However, simple removal reduces size, so we only do it if we are stuck and want to explore.
            # But if we just remove, we decrease size. We should only remove if we can immediately add back?
            # That's a swap.
            # What about Remove 2, Add 2?
            # This is more complex. Let's stick to Add and Swap for now.
            # If stuck, try a random restart or break.
            
            # If we are stuck, break out of this local search and try another start
            break
            
        return current_set, current_sums

    # Main Loop: Random Restarts
    # We start from the prime set, but also try random sets or different orderings.
    
    # Strategy 1: Local search from Primes
    # We already have best_set from primes.
    # Run local search from it
    if time.time() < deadline:
        budget = min(10.0, deadline - time.time())
        local_search(best_set, initial_sums, budget)
        
    # Strategy 2: Random Greedy + Local Search
    # Generate random permutations of 2..40 and greedily build a set, then local search.
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 50:
        attempts += 1
        candidates = list(range(2, P['n'] + 1))
        random.shuffle(candidates)
        
        current_set = []
        current_sums = {zero_vec}
        
        for c in candidates:
            ok, new_sums = can_add(current_set, current_sums, c)
            if ok:
                current_set.append(c)
                current_sums.update(new_sums)
                if len(current_set) >= P['maximum_size']:
                    break
        
        # Now local search from this greedy result
        if time.time() < deadline - 2.0:
            # Small budget for local search per restart
            ls_budget = min(2.0, deadline - time.time())
            current_set, current_sums = local_search(current_set, current_sums, ls_budget)
            
    # Final Check
    # Ensure best_set is valid
    if best_set:
        final_sums = compute_sums(best_set)
        if len(final_sums) != (1 << len(best_set)):
            # Fallback to primes if something went wrong
            best_set = primes[:]
            best_size = len(primes)
            
    # Return list of ints
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
