import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe (150s total, leave margin)
    
    required = [1, 2, 5, 11]
    max_val = 284
    required_set = set(required)
    
    # Helper to get initial sums for a set
    def get_sums(seq):
        sums = [False] * (2 * max_val + 2)
        for i in range(len(seq)):
            for j in range(i, len(seq)):
                s = seq[i] + seq[j]
                sums[s] = True
        return sums

    # Initial sums for required elements
    # Sums: 1+1=2, 1+2=3, 1+5=6, 1+11=12, 2+2=4, 2+5=7, 2+11=13, 5+5=10, 5+11=16, 11+11=22
    initial_sums = get_sums(required)
    
    # Best found set
    best_set = required[:]
    best_size = len(best_set)
    
    # Candidates pool (excluding required)
    candidates = [x for x in range(1, max_val + 1) if x not in required_set]
    
    # Strategy 1: Randomized Greedy
    # We shuffle the candidates and greedily add if valid.
    def randomized_greedy():
        curr = required[:]
        curr_sums = initial_sums[:]  # Copy the list
        
        pool = candidates[:]
        random.shuffle(pool)
        
        for x in pool:
            # Check if x can be added
            valid = True
            # Check x + y for all y in curr
            for y in curr:
                s = x + y
                if curr_sums[s]:
                    valid = False
                    break
            # Check 2*x
            if valid and curr_sums[2 * x]:
                valid = False
            
            if valid:
                curr.append(x)
                # Update sums
                for y in curr[:-1]:
                    curr_sums[x + y] = True
                curr_sums[2 * x] = True
        
        return curr

    # Strategy 2: Local Search (Kick-out)
    # Try to improve the best set by removing one non-essential element and re-filling.
    def local_search_improve(base_set):
        # base_set should be a list
        if len(base_set) <= len(required):
            return base_set
        
        # Create a list of removable elements (non-required)
        removable = [x for x in base_set if x not in required_set]
        if not removable:
            return base_set
            
        # Shuffle removable to try different kicks
        random.shuffle(removable)
        
        best_local = base_set[:]
        
        for rem_val in removable:
            # Try removing rem_val
            new_set = [x for x in base_set if x != rem_val]
            new_sums = get_sums(new_set)
            
            # Now run a fast greedy fill on the remaining candidates
            # Candidates are those not in new_set
            # We can reuse the global candidates list, filtering out those in new_set
            pool = [x for x in candidates if x not in set(new_set)]
            random.shuffle(pool)
            
            for x in pool:
                valid = True
                for y in new_set:
                    s = x + y
                    if new_sums[s]:
                        valid = False
                        break
                if valid and new_sums[2 * x]:
                    valid = False
                
                if valid:
                    new_set.append(x)
                    for y in new_set[:-1]:
                        new_sums[x + y] = True
                    new_sums[2 * x] = True
            
            if len(new_set) > len(best_local):
                best_local = new_set[:]
        
        return best_local

    # Main Loop
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Phase 1: Randomized Greedy
        curr_set = randomized_greedy()
        if len(curr_set) > best_size:
            best_size = len(curr_set)
            best_set = curr_set[:]
        
        # Phase 2: Local Search on the current best
        # We do this less frequently or only if we have time, as it's more expensive
        if iterations % 5 == 0:
            improved_set = local_search_improve(best_set)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set[:]
                
        # Early exit if we reach the theoretical max (20)
        if best_size >= 20:
            break

    return sorted(best_set)
