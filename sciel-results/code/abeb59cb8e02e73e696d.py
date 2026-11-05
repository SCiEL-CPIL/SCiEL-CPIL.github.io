"""Local search with blocked counts for large sum-free subsets."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def search_instance(elements, seed_val, time_budget):
        """Local search to find large sum-free subset."""
        rng = random.Random(seed_val)
        n = len(elements)
        elem_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Precompute all potential sum relationships
        # For each element c, list of (a, b) such that a+b=c
        # For each element a, list of (b, c) such that a+b=c
        # We need to know, for a given set S, which elements outside S are blocked.
        
        # Efficient representation:
        # blocked_by[x] = set of elements y in S such that y blocks x.
        # x is blocked if:
        # 1. There exists a, b in S such that a+b=x. (y could be a or b)
        # 2. There exists a, c in S such that x+a=c. (y could be a or c)
        # 3. 2x in S. (y is 2x)
        
        # Let's precompute for each x, the list of "conflict pairs" that block it.
        # Actually, it's easier to just check on the fly if n is small enough, 
        # but n=240 is small. O(n^2) check for blocked status is 57600 ops.
        # We can do better.
        
        # Precompute sums:
        # sum_pairs[c_val] = list of (a_val, b_val) such that a+b=c_val, a,b in elements
        # diff_pairs[c_val] = list of (a_val, x_val) such that x+a=c_val, a,x in elements? No, x is external.
        
        # Let's stick to a simpler, robust local search that is fast enough for n=240.
        # O(n^2) per iteration is ~0.05ms in Python if optimized? 
        # 240^2 = 57600. 1000 iterations = 57M ops. Might be too slow in pure Python loops.
        # We need to optimize the "is blocked" check.
        
        # Strategy: Maintain S as a set.
        # For each x not in S, determine if it can be added.
        # x can be added if:
        #   - No a, b in S such that a+b=x
        #   - No a in S such that x+a in S
        #   - 2x not in S
        
        # To speed this up, we can precompute for each x:
        # potential_blockers[x] = list of y in elements such that y interacts with x to form a sum with another element in elements.
        
        # Let's precompute:
        # For each x in elements:
        #   blockers[x] = set()
        #   For each y in elements:
        #       if x + y in elem_to_idx: # y and x sum to z in elements
        #           # Then if x and y are in S, z is blocked? No.
        #           # If z is in S, then x and y cannot both be in S.
        #           # If x is in S, y is blocked by z? No, y is blocked if x+z' = y?
        
        # Let's reframe:
        # x is blocked by y if:
        # 1. y + z = x for some z in S. (Here y is a blocker, z is also a blocker)
        # 2. x + y = z for some z in S. (Here y is a blocker, z is also a blocker)
        # 3. 2y = x? No, 2x = y. If y=2x is in S, x is blocked.
        
        # So for each x, we can precompute the set of "partner" elements that, if present in S along with another specific element, block x.
        # Actually, it's simpler to just iterate through S to check if x is blocked.
        # Since n=240, |S| is at most 80 (Bourgain bound ~ (n+2)/3).
        # Checking if x is blocked:
        #   for a in S:
        #       if x - a in S: return True # x = a + (x-a)
        #       if x + a in S: return True # x + a = (x+a)
        #   if 2*x in S: return True
        # This is O(|S|) per x.
        # Total check for all x in V\S is O(n * |S|) ~ 240 * 80 = 19200 ops.
        # This is very fast! We can do thousands of iterations.
        
        def is_blocked(x, S_set):
            # Check if x can be added to S_set without violation
            # Violation if exists a, b in S_set: a+b=x
            # Or exists a, c in S_set: x+a=c
            # Or 2x in S_set
            
            # Check 1: x = a + b
            # We can just iterate a in S_set, check if x-a in S_set
            for a in S_set:
                if x - a in S_set:
                    return True
                if x + a in S_set:
                    return True
            if 2 * x in S_set:
                return True
            return False

        # Initial solution: Greedy
        # Sort elements by "conflict potential" (number of sums/diffs with other elements)
        # Or just random.
        
        # Let's try multiple random initializations and pick the best.
        best_size = 0
        best_S = set()
        
        num_restarts = 20
        # Estimate time per restart
        # Each restart: 
        # 1. Init O(n)
        # 2. Repair O(n * |S|) * n? 
        #    Repair: while violations exist, remove one.
        #    To find a violation, we can just check all pairs? O(|S|^2).
        #    Better: maintain S, and just try to build it greedily.
        
        def greedy_build(elements, rng):
            S = set()
            # Random order
            elems = elements[:]
            rng.shuffle(elems)
            for x in elems:
                if not is_blocked(x, S):
                    S.add(x)
            return S

        # Local Search on a valid S
        def local_search(S, time_limit):
            S = set(S)
            S_set = S
            n_local = len(elements)
            
            # Precompute indices for speed? Not needed if using set.
            
            while True:
                t_start = time.time()
                if t_start > time_limit:
                    break
                
                improved = False
                # Try to add any unblocked element
                # To be efficient, pick a random candidate from V\S
                # Or iterate through all? n=240 is small.
                
                # 1. Try adding
                for x in elements:
                    if x not in S_set:
                        if not is_blocked(x, S_set):
                            S_set.add(x)
                            improved = True
                            break # Restart check after one add? Or continue?
                
                if improved:
                    continue # Restart the loop to check more adds
                
                # 2. If no add possible, try 2-swap
                # Remove y, then try to add as many as possible.
                # We want to find a y such that removing it allows adding >1 elements.
                
                # Heuristic: Pick y that is "central" (blocks many elements)
                # For each y in S, count how many x in V\S are blocked ONLY by y (or become unblocked if y is removed).
                
                # This is expensive to compute exactly.
                # Alternative: Random 2-swap.
                # Pick random y in S.
                # Remove y.
                # Try to add any unblocked x.
                # If we added k elements, and k > 1, keep it.
                # If k == 1, it's a wash (size same), but might lead to better config. Accept with prob?
                # If k == 0, revert.
                
                if len(S_set) == 0:
                    break
                    
                y = rng.choice(list(S_set))
                S_set.remove(y)
                
                added_count = 0
                added_elems = []
                for x in elements:
                    if x not in S_set and x != y:
                        if not is_blocked(x, S_set):
                            S_set.add(x)
                            added_count += 1
                            added_elems.append(x)
                            # Don't break, try to add all possible?
                            # This is O(n * |S|) for each x. 
                            # If we do this for all x, it's O(n^2 * |S|).
                            # For n=240, |S|=80, 240*240*80 = 4.6M ops. 
                            # This might be too slow for many iterations.
                            # Let's just try to add ONE element to see if it's worth it.
                            # If we can add at least one, it's a valid move (size doesn't decrease).
                            # But we want to increase size.
                            # So we need to add at least 2 to increase size by 1 (since we removed 1).
                            
                            # Let's just try to add the first one we find.
                            break 
                            
                if added_count > 1:
                    # Improved size
                    pass
                elif added_count == 1:
                    # Same size, different composition. Accept with probability 0.5?
                    if rng.random() > 0.5:
                        pass # Keep
                    else:
                        S_set.remove(added_elems[0])
                        S_set.add(y)
                else:
                    # Revert
                    S_set.add(y)

            return S_set

        # Main loop
        for r in range(num_restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            # Allocate time per restart
            remaining = DEADLINE - (time.time() - start_time)
            if remaining < 1:
                break
            t_limit = time.time() + remaining / (num_restarts - r)
            
            S_init = greedy_build(elements, rng)
            S_res = local_search(S_init, t_limit)
            
            if len(S_res) > best_size:
                best_size = len(S_res)
                best_S = S_res
                
        return sorted(list(best_S))

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))
    
    results = []
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        
        # Allocate time budget: more time for larger instances
        remaining = DEADLINE - (time.time() - start_time)
        if i == 0:
            time_budget = remaining * 0.2
        elif i == 1:
            time_budget = remaining * 0.3
        else:
            time_budget = remaining * 0.5
        
        result = search_instance(elems, seed_val=801 + i, time_budget=time_budget)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
