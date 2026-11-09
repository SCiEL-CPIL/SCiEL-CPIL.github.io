import itertools
import random
import time

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0   # Safety margin under the 90s kill
    N = 4096

    # Helper to check if v can be added to chosen set with existing sums
    def check_add(v, chosen, sums):
        """
        Checks if v can be added to the set 'chosen' without creating duplicate triple sums.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        new_sums = set()
        # Triple (v, v, v)
        new_sums.add(3 * v)
        
        # Triples (v, v, u) and (v, u, u) for u in chosen
        for u in chosen:
            new_sums.add(2 * v + u)
            new_sums.add(v + 2 * u)
            
        # Triples (v, u, w) for u, w in chosen with u <= w
        for i in range(len(chosen)):
            u = chosen[i]
            for j in range(i, len(chosen)):
                w = chosen[j]
                new_sums.add(v + u + w)
                
        # Check for conflicts with existing sums
        if new_sums & sums:
            return False, None
        return True, new_sums

    # Full validation for safety
    def is_valid(vals):
        m = len(vals)
        seen = set()
        for i in range(m):
            a = vals[i]
            for j in range(i, m):
                b = vals[j]
                for k in range(j, m):
                    c = vals[k]
                    s = a + b + c
                    if s in seen:
                        return False
                    seen.add(s)
        return True

    # Greedy construction from a specific order of candidates
    def greedy_build(order, limit):
        chosen = []
        sums = set()
        for v in order:
            if time.time() > limit:
                break
            ok, ns = check_add(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
        return chosen

    # Local improvement via swapping
    def local_improve_swap(chosen, limit):
        """
        Tries to improve the set by swapping elements.
        1. Try to replace a small element with a larger one if valid.
        2. Re-run greedy fill after a successful swap.
        """
        current_set = set(chosen)
        best_size = len(chosen)
        best_set = list(chosen)
        
        # We try to swap out elements from the end (largest) or start (smallest)?
        # Usually, removing a "blocking" element allows more.
        # Let's try removing each element and seeing if we can add more.
        
        # To keep it fast, we only do a few passes.
        for _ in range(5): # Max 5 swap attempts
            if time.time() > limit:
                break
            
            improved = False
            # Try removing each element
            for i in range(len(chosen)):
                if time.time() > limit:
                    break
                u = chosen[i]
                # Tentatively remove u
                trial_set = chosen[:i] + chosen[i+1:]
                trial_sums = set()
                for x in itertools.combinations_with_replacement(trial_set, 3):
                    trial_sums.add(sum(x))
                
                # Try to add any v not in trial_set
                added_any = False
                for v in range(1, N + 1):
                    if time.time() > limit:
                        break
                    if v in trial_set:
                        continue
                    ok, ns = check_add(v, trial_set, trial_sums)
                    if ok:
                        trial_set.append(v)
                        trial_sums |= ns
                        added_any = True
                        # Stop adding if we've reached a good size or just do one pass?
                        # Doing a full greedy fill here is expensive. 
                        # Let's just check if we can add *at least one* to make it better?
                        # Actually, if we remove 1 and add 1, size is same.
                        # We want to increase size.
                        # So if we remove u, we must be able to add >1 elements to beat current size?
                        # Or just add 1 element that is "better" (larger)?
                        # For maximizing cardinality, we just need size to increase.
                        # If we remove 1, we need to add >= 2 to increase size? No, if we add 2, size increases by 1.
                        
                        # This inner loop is O(N * M^2). Too slow for 4096 * 30^2 * 30 swaps.
                        # We need a faster check.
                        pass
                
                # This approach is too slow. Let's stick to randomized restarts which are faster.
                break # Skip this complex local search for now, rely on restarts
                
        return best_set

    best = []
    
    # Strategy 1: Powers of 3 core + Greedy
    pow3 = [3 ** i for i in range(0, 8) if 3 ** i <= N]
    if is_valid(pow3):
        # Extend powers of 3 greedily
        order = list(range(1, N + 1))
        # Ensure pow3 elements are prioritized? 
        # Greedy build will pick them if they are in order.
        # But we want to start with them.
        chosen = list(pow3)
        sums = set()
        for x in itertools.combinations_with_replacement(chosen, 3):
            sums.add(sum(x))
            
        for v in range(1, N + 1):
            if time.time() > DEADLINE - 10:
                break
            if v in chosen:
                continue
            ok, ns = check_add(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
        
        if is_valid(chosen) and len(chosen) > len(best):
            best = chosen

    # Strategy 2: Randomized Greedy Restarts
    random.seed(42)
    
    # We will run multiple trials with different random orderings
    num_trials = 100
    trial_time = (DEADLINE - 5) / num_trials
    
    for t in range(num_trials):
        if time.time() > DEADLINE - 5:
            break
        
        # Generate a random order
        # Mix of small numbers and random numbers
        # Option A: Pure random permutation
        # Option B: Sorted, but with random swaps
        # Option C: Random weights
        
        # Let's try a "Randomized Ascending" approach:
        # Pick the next candidate from the smallest K available, with some probability picking a larger one.
        
        # Simpler: Random permutation of 1..N
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Bias towards smaller numbers?
        # Sort by (random() * 1000 + v) ?
        
        # Let's just use a shuffled list.
        # To improve, we can try "Greedy with Randomized Tie-Breaking"
        # Or "Greedy from Random Subsets"
        
        # Try 1: Pure Random
        res = greedy_build(order, DEADLINE - 5)
        if is_valid(res) and len(res) > len(best):
            best = res
            
        # Try 2: Random but biased to small
        # Create a list where small numbers appear more often?
        # No, greedy takes first valid.
        # Let's try a different order: Sort by v * random()
        order2 = list(range(1, N + 1))
        keys = [v * random.uniform(0.5, 2.0) for v in order2]
        order2 = [v for v, k in sorted(zip(order2, keys))]
        
        if time.time() < DEADLINE - 5:
            res2 = greedy_build(order2, DEADLINE - 5)
            if is_valid(res2) and len(res2) > len(best):
                best = res2

    # Strategy 3: Local Search on Best
    # Try to add more to the best set found
    if time.time() < DEADLINE - 2:
        chosen = list(best)
        sums = set()
        for x in itertools.combinations_with_replacement(chosen, 3):
            sums.add(sum(x))
        
        # Try to add any missing number
        for v in range(1, N + 1):
            if time.time() > DEADLINE - 1:
                break
            if v in chosen:
                continue
            ok, ns = check_add(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
        
        if is_valid(chosen) and len(chosen) > len(best):
            best = chosen

    # Final sanity check
    best = sorted(set(best))
    return best
