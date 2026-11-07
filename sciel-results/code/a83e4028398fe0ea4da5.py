import random
import time

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute exponent vectors
    vecs = {}
    for n in range(2, 41):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[n] = tuple(v)
        
    zero = (0,) * num_primes
    best_set = []
    best_len = 0
    
    # Initial good set: primes
    initial_set = primes[:]
    # Verify and compute sums for initial set
    sums = {zero}
    valid_initial = True
    for p in initial_set:
        v = vecs[p]
        new_sums = []
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            if ns in sums:
                valid_initial = False
                break
            new_sums.append(ns)
        if not valid_initial:
            break
        sums.update(new_sums)
        
    if valid_initial:
        best_set = initial_set
        best_len = len(initial_set)

    start_time = time.time()
    deadline = start_time + 50
    
    def local_search(current_set, sums, remaining_time_budget):
        """Try to improve current_set via add, remove, or swap operations."""
        nonlocal best_set, best_len
        current_len = len(current_set)
        in_set = set(current_set)
        
        # Try adding elements
        for n in range(2, 41):
            if n in in_set:
                continue
            v = vecs[n]
            collision = False
            new_sums_list = []
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(num_primes))
                if ns in sums:
                    collision = True
                    break
                new_sums_list.append(ns)
            
            if not collision:
                new_set = current_set + [n]
                new_sums = sums.copy()
                new_sums.update(new_sums_list)
                if len(new_set) > best_len:
                    best_len = len(new_set)
                    best_set = new_set
                    current_set = new_set
                    sums = new_sums
                    in_set.add(n)
                    current_len += 1
                if time.time() > start_time + remaining_time_budget:
                    return current_set, sums
        
        # Try swaps: remove one, add another
        for i in range(len(current_set)):
            removed = current_set[i]
            # Remove from sums
            v_removed = vecs[removed]
            new_sums_without = set()
            for s in sums:
                # Can't easily remove; rebuild from scratch is expensive
                # Instead, try adding candidates to see if we can replace
                pass
            
            # Try adding each candidate not in set
            for n in range(2, 41):
                if n in in_set:
                    continue
                v_add = vecs[n]
                # Check if current_set minus removed plus n is valid
                temp_set = current_set[:i] + current_set[i+1:] + [n]
                temp_sums = {zero}
                valid = True
                for elem in temp_set:
                    v = vecs[elem]
                    new_sums_temp = []
                    for s in temp_sums:
                        ns = tuple(s[j] + v[j] for j in range(num_primes))
                        if ns in temp_sums:
                            valid = False
                            break
                        new_sums_temp.append(ns)
                    if not valid:
                        break
                    temp_sums.update(new_sums_temp)
                
                if valid and len(temp_set) > best_len:
                    best_len = len(temp_set)
                    best_set = temp_set
                    current_set = temp_set
                    sums = temp_sums
                    in_set = set(current_set)
                    current_len = len(current_set)
                    return current_set, sums  # Restart local search with improved set
                
                if time.time() > start_time + remaining_time_budget:
                    return current_set, sums
        
        return current_set, sums

    # Random restarts
    while time.time() < deadline:
        candidates = list(range(2, 41))
        random.shuffle(candidates)
        
        sums = {zero}
        current_set = []
        
        for n in candidates:
            v = vecs[n]
            collision = False
            new_sums_list = []
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(num_primes))
                if ns in sums:
                    collision = True
                    break
                new_sums_list.append(ns)
            
            if not collision:
                sums.update(new_sums_list)
                current_set.append(n)
        
        # Apply local search to improve
        if len(current_set) >= best_len:
            remaining = max(0, deadline - time.time())
            current_set, sums = local_search(current_set, sums, remaining * 0.3)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set
            
    return best_set
