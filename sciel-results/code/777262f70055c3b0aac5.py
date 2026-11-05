"""WalkSAT-style local search for large sum-free subsets."""
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
        """
        WalkSAT-inspired local search for sum-free subsets.
        State: mask (list of 0/1)
        Objective: Maximize size subject to sum-free constraint.
        Strategy:
        1. Initialize with a random subset.
        2. Local Search: Flip bits to reduce violations. If stuck (no single flip reduces violations),
           perform a 'walk' by flipping a random variable involved in a violation.
        3. Greedy Maximization: Once sum-free, try to add more elements.
        """
        rng = random.Random(seed_val)
        n = len(elements)
        elem_set = set(elements)
        
        # Precompute sums for O(1) violation check? 
        # For n=240, O(n^2) per check is 57600 ops. 
        # In Python, this is slow. We need incremental updates or a faster check.
        
        # Let's use a set for the current subset S.
        # Violation count: number of pairs (a,b) in S such that a+b in S.
        # Note: a=b is allowed.
        
        def get_violation_count(S_set):
            """Count total violations in S_set."""
            count = 0
            # To avoid double counting (a,b) and (b,a), we could iterate uniquely, 
            # but for the heuristic, total count is fine.
            # Actually, for WalkSAT, we need to know which variables are involved in violations.
            # Let's just compute the set of violating elements.
            viol_elements = set()
            # Optimization: iterate through elements, check if x = a+b for a,b in S.
            # Or for each pair (a,b), check if a+b in S.
            # n=240 -> 240*240/2 = 28800 pairs.
            # Checking if a+b in S is O(1).
            # So 28800 ops per full evaluation.
            # This is feasible for a few thousand iterations.
            
            for a in S_set:
                for b in S_set:
                    if a + b in S_set:
                        viol_elements.add(a)
                        viol_elements.add(b)
                        viol_elements.add(a+b)
            return len(viol_elements), viol_elements

        # Initialize
        # Start with a random subset, maybe biased towards "safe" numbers?
        # Numbers with same sign are safer? No, -1 + -1 = -2.
        # Let's just start random.
        mask = [1 if rng.random() < 0.5 else 0 for _ in range(n)]
        current_S = set(elements[i] for i in range(n) if mask[i])
        
        # Ensure we have a valid starting point? No, WalkSAT handles violations.
        
        best_size = 0
        best_mask = mask[:]
        
        # If initial is sum-free, record
        if not current_S:
             pass
        else:
             # Quick check
             is_viol = False
             S_list = list(current_S)
             for a in S_list:
                 for b in S_list:
                     if a + b in current_S:
                         is_viol = True
                         break
                 if is_viol: break
             if not is_viol:
                 best_size = len(current_S)
        
        time_limit = start_time + time_budget
        
        # WalkSAT Loop
        # We maintain:
        # - mask
        # - current_S (set of values)
        # - viol_elements (set of values involved in violations)
        # - total_viol_count (maybe not needed if we just use viol_elements)
        
        # To speed up, we can cache violation info? 
        # Let's try a simpler approach first: 
        # 1. Compute viol_elements
        # 2. If empty, we are sum-free. Try to add.
        # 3. If not empty, pick a variable to flip.
        #    - With prob p (e.g., 0.7), pick a variable that reduces violations (if any exist).
        #      To find this, we'd need to evaluate delta for each flip. O(n * n^2) = O(n^3) too slow.
        #    - Alternative: Just pick a random variable from viol_elements and flip it.
        #      This is the "walk" part. The "local" part is harder without gradient.
        #      
        # Let's use a hybrid:
        # - Compute viol_elements.
        # - If viol_elements is empty: Greedy add.
        # - Else: 
        #   - Identify "critical" variables: those in viol_elements.
        #   - Randomly pick one from viol_elements.
        #   - Flip it.
        #   - Update current_S.
        #   - Recompute viol_elements (expensive, O(n^2)).
        
        # To make it faster, we can limit the number of iterations.
        # n=240, O(n^2) = 57k ops. Python does ~10M-50M simple ops/sec? 
        # Set operations are slower. Let's say 10k-50k iterations possible in 10s.
        
        max_iters = 20000
        it = 0
        last_improve = 0
        stagnant_count = 0
        
        while it < max_iters:
            if time.time() > time_limit:
                break
            
            # Compute violations
            # Optimization: If current_S is small, this is fast.
            viol_elements = set()
            S_list = list(current_S)
            # Early exit if possible? No, need full set for greedy later.
            for a in S_list:
                for b in S_list:
                    s = a + b
                    if s in current_S:
                        viol_elements.add(a)
                        viol_elements.add(b)
                        viol_elements.add(s)
                        
            if not viol_elements:
                # Sum-free!
                # Try to add more elements greedily
                added = True
                while added:
                    added = False
                    # Shuffle elements to avoid bias?
                    # Try adding elements not in current_S
                    for x in elements:
                        if x not in current_S:
                            # Check if adding x creates violation
                            # x = a+b, a+b=x, a+x=b, b+x=a
                            # Since we check all pairs in S, adding x means:
                            # 1. x = a+b for a,b in S
                            # 2. a+x = b for a,b in S (implies b-a=x)
                            # 3. b+x = a for a,b in S (implies a-b=x)
                            # 4. x+x = y for y in S (implies 2x in S)
                            
                            can_add = True
                            # Check if x is sum of two in S
                            # This is O(n^2) if done naively.
                            # We can check: for a in S, if x-a in S.
                            for a in S_list: # S_list is outdated if we added, but we recompute loop
                                if (x - a) in current_S:
                                    can_add = False
                                    break
                            if can_add:
                                # Check if a+x in S or x+x in S
                                # Actually, if x-a in S, then a+(x-a)=x.
                                # We also need to check if a+x is in S.
                                # If a+x = y in S, then y-x=a.
                                # So checking (x-a) in S covers a+x=y? No.
                                # a+x=y => y-x=a.
                                # So we need to check if (y-x) in S for any y in S.
                                # Which is equivalent to: for y in S, if y-x in S.
                                # This is the same as: for a in S, if a+x in S.
                                
                                # Let's just do a full check for safety, it's O(n)
                                for y in S_list:
                                    if (y + x) in current_S or (y - x) in current_S or (x - y) in current_S:
                                        # Wait, if y+x in S, violation.
                                        # If y-x in S, then y = (y-x)+x, violation.
                                        # If x-y in S, then x = y+(x-y), violation.
                                        can_add = False
                                        break
                                # Also check 2x in S
                                if can_add and (2*x) in current_S:
                                    can_add = False
                                    
                            if can_add:
                                current_S.add(x)
                                mask[elements.index(x)] = 1 # This is O(n), bad. Use dict?
                                added = True
                                S_list = list(current_S) # Update list
                                break # Restart loop to be safe with S_list update
                
                # Update best
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_mask = mask[:]
                    last_improve = it
                    stagnant_count = 0
                else:
                    stagnant_count += 1
                    # If stagnant, maybe restart or continue?
                    # Let's continue, maybe a different walk helps.
                    
                # Restart with a slightly different mask? 
                # Or just continue searching for larger sets?
                # Since we are sum-free, we can try to flip some 1s to 0s to allow adding more 1s?
                # That's complex. Let's just do a random walk for a bit to escape this local optimum.
                
            else:
                # Violations exist
                # Pick a random element from viol_elements to flip
                # We need to map value back to index
                # We have `elements` list.
                # We need a value->index map for O(1) lookup.
                # Let's create it once outside the loop? 
                # But `elements` is fixed.
                
                # To avoid O(n) lookup for index, let's maintain a dict.
                # But we are inside search_instance.
                
                # Let's pick a random value from viol_elements
                val_to_flip = rng.choice(list(viol_elements))
                
                # Find index
                # Linear search is O(n). n=240. 240*20000 = 4.8M ops. Acceptable.
                idx = -1
                for i, v in enumerate(elements):
                    if v == val_to_flip:
                        idx = i
                        break
                
                if idx != -1:
                    # Flip
                    mask[idx] = 1 - mask[idx]
                    if mask[idx]:
                        current_S.add(val_to_flip)
                    else:
                        current_S.remove(val_to_flip)
                    
                it += 1
                if it % 100 == 0:
                    # Check time more frequently?
                    pass
        
        # Final Greedy Pass on best_mask
        result_set = set(elements[i] for i in range(n) if best_mask[i])
        # Try to add more
        # We do a few passes of greedy addition
        for _ in range(5):
            changed = False
            for x in elements:
                if x not in result_set:
                    can_add = True
                    # Check if x creates violation
                    # x = a+b
                    for a in result_set:
                        if (x - a) in result_set:
                            can_add = False
                            break
                    if can_add:
                        # Check if a+x in result_set or x+x in result_set
                        for y in result_set:
                            if (y + x) in result_set or (2*x) in result_set:
                                can_add = False
                                break
                    if can_add:
                        result_set.add(x)
                        changed = True
            if not changed:
                break
        
        return list(result_set)

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))
    
    results = []
    total_time = 0
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        
        # Allocate time budget
        # Larger instances need more time
        if i == 0:
            time_budget = 15
        elif i == 1:
            time_budget = 30
        else:
            time_budget = 90
            
        # Ensure we don't exceed deadline
        if time.time() + time_budget > DEADLINE:
            time_budget = DEADLINE - time.time() - 1
            
        if time_budget < 1:
             # Fallback
             results.append([x for x in elems if x % 2 != 0])
             continue

        result = search_instance(elems, seed_val=801 + i, time_budget=time_budget)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
