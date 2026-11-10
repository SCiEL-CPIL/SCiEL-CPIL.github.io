"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute all possible sums of 3 distinct numbers from 1..192
    # This allows O(1) lookup for "does sum S exist in the current set?"
    # Sums range from 1+2+3=6 to 190+191+192=573
    MAX_SUM = 190 + 191 + 192
    sum_counts = [0] * (MAX_SUM + 1)
    
    def check_valid(selected, new_val):
        """
        Checks if adding new_val to selected maintains the property.
        We only need to check:
        1. Does new_val divide any sum of 3 distinct OTHERS in (selected + [new_val])?
           Others are all elements in selected.
           So we check if sum_counts[s] >= 3 for any s in {3*new_val, 4*new_val, ...}
           Wait, sum_counts counts subsets of the CURRENT selected set.
           If we are adding new_val, the "others" for new_val are exactly the current selected set.
           So we just need to know if any sum of 3 distinct elements from selected is divisible by new_val.
           
        2. For each a in selected, does a divide any sum of 3 distinct OTHERS in (selected + [new_val])?
           The others for a are (selected \ {a}) U {new_val}.
           Previously, we knew that for all a in selected, a did NOT divide any sum of 3 distinct elements from (selected \ {a}).
           Now, the new possible sums of 3 distinct elements from (selected \ {a}) U {new_val} are:
           - Sums of 3 elements from (selected \ {a}) (already known safe)
           - Sums of 2 elements from (selected \ {a}) + new_val
           
           So we need to check: for each a in selected, is there any pair (u, v) in selected (u != v, u != a, v != a) 
           such that (u + v + new_val) % a == 0?
        """
        # Check 1: new_val as the divisor
        # We need to check if any sum of 3 distinct elements from 'selected' is divisible by new_val.
        # We can precompute the possible sums of 3 distinct elements from 'selected'.
        # However, doing this from scratch every time is O(N^3).
        # Let's maintain sum_counts for the current 'selected'.
        
        # Actually, maintaining sum_counts incrementally is O(N^2) per insertion.
        # Let's re-evaluate the cost.
        # N is small (max 48). N^2 is ~2300.
        # For each candidate, we do O(N^2) to update counts and O(N^2) to check pairs?
        
        # Let's implement the check directly without precomputing global sum_counts for the whole universe, 
        # but rather for the current set, or just compute on the fly if N is small.
        # Given N <= 48, O(N^3) is 48^3 = 110k operations. 
        # If we do this for 192 candidates, that's 21M ops. Python might be slow.
        
        # Better approach: Maintain sum_counts for the current 'selected' set.
        # When we add x:
        # 1. Check if x divides any sum of 3 from current selected.
        #    We can check this by iterating multiples of x up to MAX_SUM and checking if sum_counts[multiple] > 0.
        #    But wait, sum_counts[s] counts the number of pairs? No, it counts the number of triples?
        #    If we maintain sum_counts[s] = number of distinct triples in 'selected' that sum to s, 
        #    updating it when adding x requires adding x to all pairs in 'selected'.
        #    Pairs in 'selected': O(N^2).
        #    So update is O(N^2).
        #    Check for x: iterate multiples of x. O(MAX_SUM/x).
        
        # 2. Check for existing a in selected:
        #    Does a divide (u + v + x) for any distinct u, v in selected, u,v != a?
        #    This is equivalent to: does there exist u, v in selected (u != v, u != a, v != a) such that (u+v) % a == (-x) % a?
        #    We can precompute pair_sums_mod_a? That's complex.
        #    Simpler: Just iterate u, v in selected. O(N^2) per a. Total O(N^3).
        #    With N=48, N^3 = 110k. This is fast enough in Python if optimized.
        
        # Let's try a hybrid: 
        # Maintain sum_counts for triples in 'selected'.
        # To check if adding x is valid:
        # 1. Check if any triple sum in 'selected' is divisible by x.
        #    Iterate m = x, 2x, ... up to MAX_SUM. If sum_counts[m] > 0, invalid.
        # 2. For each a in selected:
        #    We need to check if there are u, v in selected (u != a, v != a, u != v) such that (u + v + x) % a == 0.
        #    This is (u + v) % a == (a - x % a) % a.
        #    We can precompute for each a, a count of pair sums modulo a?
        #    Or just brute force u, v for each a. 
        #    Since N is small, let's just brute force the pair check for existing elements.
        
        # Let's refine the "Brute Force" check to be as fast as possible.
        
        # Check 1: x divides sum of 3 from selected
        # We need to know if any triple in 'selected' sums to a multiple of x.
        # If we maintain sum_counts, this is fast.
        
        # Check 2: For each a in selected, does a divide (u+v+x) for some distinct u,v in selected \ {a}?
        # Let target = (a - x % a) % a.
        # We need to know if there are distinct u, v in selected, u != a, v != a, such that (u + v) % a == target.
        
        # This suggests we should maintain for each 'a' in the set, a frequency array of pair sums modulo a?
        # That's too much state.
        
        # Alternative: Just do the O(N^3) check for the new element x against the existing set.
        # O(N^3) with N=48 is ~100k ops. 
        # If we try 192 candidates, 192 * 100k = 19M ops.
        # In Python, 19M simple ops might take 1-2 seconds. 
        # We have 85 seconds. We can afford multiple restarts or local search steps.
        
        # Let's implement a clean O(N^3) check function.
        
        # For x (new element):
        # 1. Check if x divides sum of any 3 distinct from selected.
        #    for i in range(len(selected)):
        #       for j in range(i+1, len(selected)):
        #          for k in range(j+1, len(selected)):
        #             if (selected[i]+selected[j]+selected[k]) % x == 0: return False
        # 2. For each a in selected:
        #    Check if a divides (u+v+x) for any distinct u,v in selected \ {a}.
        #    for i in range(len(selected)):
        #       u = selected[i]
        #       if u == a: continue
        #       for j in range(len(selected)):
        #          v = selected[j]
        #          if v == a or v == u: continue
        #          if (u + v + x) % a == 0: return False
          
        # This is O(N^3) for part 1 and O(N^3) for part 2. Total O(N^3).
        
        # Optimization:
        # For Part 1, we can precompute all triple sums of 'selected' and check divisibility by x.
        # But 'selected' changes.
        
        # Let's just write the efficient nested loops.
        
        n = len(selected)
        
        # Part 1: x divides sum of 3 distinct from selected
        # We can optimize this by checking multiples of x.
        # But we don't have sum_counts yet. Let's build it?
        # No, let's just iterate.
        
        # Actually, we can maintain sum_counts incrementally.
        # Let's pass sum_counts to the function and update it if valid.
        
        # Check Part 1:
        # If we have sum_counts for 'selected', we just check if any multiple of x has count > 0.
        for m in range(x, MAX_SUM + 1, x):
            if sum_counts[m] > 0:
                return False, sum_counts
        
        # Check Part 2:
        # For each a in selected, check if there exist u, v in selected (distinct, not a) s.t. (u+v+x) % a == 0.
        for idx_a, a in enumerate(selected):
            target = (a - (x % a)) % a
            # We need distinct u, v from selected, u != a, v != a.
            # We can iterate u, v.
            # To avoid u==v and u==a or v==a.
            # Since N is small, just double loop.
            for i in range(n):
                u = selected[i]
                if u == a:
                    continue
                for j in range(i + 1, n): # ensure u != v by index
                    v = selected[j]
                    if v == a:
                        continue
                    if (u + v) % a == target:
                        return False, sum_counts
                        
        # If we get here, it's valid.
        # Now update sum_counts for the new set selected + [x].
        # New triples are:
        # 1. Old triples (already in sum_counts)
        # 2. Triples containing x: x + u + v where u, v are distinct from selected.
        
        new_sum_counts = sum_counts[:] # Copy is O(MAX_SUM) ~ 600. Cheap.
        
        for i in range(n):
            u = selected[i]
            for j in range(i + 1, n):
                v = selected[j]
                s = x + u + v
                new_sum_counts[s] += 1
                
        return True, new_sum_counts

    def local_search(initial_set, time_limit):
        selected = list(initial_set)
        # Rebuild sum_counts from scratch for this set
        sum_counts = [0] * (MAX_SUM + 1)
        n = len(selected)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    s = selected[i] + selected[j] + selected[k]
                    sum_counts[s] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding any number not in selected
            # Sort candidates by value? Maybe larger numbers are safer?
            # Let's try all candidates.
            candidates = [x for x in range(1, 193) if x not in selected]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if time.time() > time_limit:
                    break
                valid, new_sc = check_valid(selected, x)
                if valid:
                    selected.append(x)
                    sum_counts = new_sc
                    improved = True
                    added = True
                    break # Restart loop with new set
            
            if not added:
                # Try swaps
                # Pick a random element to remove, try to add a better one?
                # Or just random swap: remove x, add y.
                # Since we want to maximize size, swaps only help if we can add more later?
                # Actually, if we can't add anything, we are stuck.
                # Swapping a small element for a large one might open up space?
                # Not necessarily. 
                # Let's try a perturbation: remove a random element, then try to add.
                if len(selected) > 0:
                    idx = random.randint(0, len(selected) - 1)
                    removed = selected[idx]
                    selected.pop(idx)
                    
                    # Rebuild sum_counts
                    sum_counts = [0] * (MAX_SUM + 1)
                    n = len(selected)
                    for i in range(n):
                        for j in range(i + 1, n):
                            for k in range(j + 1, n):
                                s = selected[i] + selected[j] + selected[k]
                                sum_counts[s] += 1
                    # Try to add something back or something new
                    # This effectively resets the "stuck" state.
                    improved = True # Force another iteration of the while loop
                    # But we didn't improve size. 
                    # To avoid infinite loops, we should limit perturbations.
                    # Let's just break if we can't add.
                    # Actually, let's just return the best found so far.
                    # The "add" phase is the main driver.
                    # If we can't add, we are at a local optimum for this set.
                    # Let's break.
                    break

        return selected

    best_set = []
    
    # Strategy 1: Greedy descending
    selected = []
    sum_counts = [0] * (MAX_SUM + 1)
    for x in range(192, 0, -1):
        valid, new_sc = check_valid(selected, x)
        if valid:
            selected.append(x)
            sum_counts = new_sc
    if len(selected) > len(best_set):
        best_set = selected[:]
        
    # Strategy 2: Local Search from Greedy Descending
    if time.time() < deadline - 10:
        ls_set = local_search(best_set, deadline - 10)
        if len(ls_set) > len(best_set):
            best_set = ls_set[:]
            
    # Strategy 3: Randomized Greedy (multiple restarts)
    restarts = 20
    for r in range(restarts):
        if time.time() > deadline - 5:
            break
        
        # Random order, biased towards larger numbers?
        # Just random shuffle of 1..192
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        selected = []
        sum_counts = [0] * (MAX_SUM + 1)
        for x in candidates:
            valid, new_sc = check_valid(selected, x)
            if valid:
                selected.append(x)
                sum_counts = new_sc
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            
        # Apply local search to this random result?
        if time.time() < deadline - 5:
            ls_set = local_search(selected, deadline - 5)
            if len(ls_set) > len(best_set):
                best_set = ls_set[:]

    return best_set
# EVOLVE-BLOCK-END
