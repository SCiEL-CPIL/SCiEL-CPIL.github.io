import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Helper: Check if adding `x` to set `S` (sorted list) is valid.
    # S is a sorted list of existing elements.
    def can_add(S, x):
        n = len(S)
        if n < 3:
            return True
        
        # Check 1: x divides sum of any 3 distinct elements in S
        # Max sum of 3 in S
        max_sum_3 = S[-1] + S[-2] + S[-3]
        if x <= max_sum_3:
            # Iterate triples
            # Optimization: if x is large, few multiples.
            # If x is small, many triples.
            # We just iterate all triples and check mod.
            # For n=40, triples ~ 10k. 10k mod ops is fast.
            for i in range(n - 2):
                xi = S[i]
                # Early break if smallest sum > max multiple? No.
                # We can break inner loops if sum > max multiple of x?
                # Max multiple is max_sum_3 // x
                # Actually, just check (xi + S[j] + S[k]) % x == 0
                for j in range(i + 1, n - 1):
                    xj = S[j]
                    # If xi + xj + S[j+1] > max_multiple * x, we might break?
                    # max_multiple * x = (max_sum_3 // x) * x
                    # This bound is loose.
                    for k in range(j + 1, n):
                        if (xi + xj + S[k]) % x == 0:
                            return False
        
        # Check 2: Any a in S divides x + y + z (y,z in S, y!=z, y,z!=a)
        # For each a in S:
        # We need (x + y + z) % a == 0 => y + z % a == (-x) % a
        # Let target_res = (-x) % a
        # We need distinct y, z in S\{a} such that (y+z) % a == target_res
        
        for idx_a, a in enumerate(S):
            # If a is very large, a > x + max_y + max_z, then only possible if sum is 0 (impossible for positive ints)
            # Max sum = x + S[-1] + S[-2]
            max_possible_sum = x + S[-1] + S[-2]
            if a > max_possible_sum:
                continue
            
            # Check if there exist y, z in S \ {a} such that (y+z) % a == (-x) % a
            # And y != z, y != a, z != a
            
            # To speed this up, we can precompute the set of residues of pair sums?
            # No, a changes.
            
            # Iterate pairs y, z in S excluding a
            # Optimization: 
            # If n is small, just do it.
            # To avoid index confusion, we can create a list of indices excluding idx_a?
            # That's O(n) overhead.
            # Alternative: Iterate all pairs (i,j) in S, skip if i==idx_a or j==idx_a.
            
            target_res = (-x) % a
            
            # We can break early if found.
            # Optimization: If a is large, few residues match.
            # If a is small, many.
            
            for i in range(n):
                if i == idx_a: continue
                yi = S[i]
                for j in range(i + 1, n):
                    if j == idx_a: continue
                    yj = S[j]
                    if ((yi + yj) % a) == target_res:
                        return False
        
        return True

    # Main ILS Loop
    rng = random.Random(42)
    best_size = 0
    best_set = set()
    
    # Initial Greedy: Descending order
    curr_set = []
    for x in range(192, 0, -1):
        if can_add(curr_set, x):
            curr_set.append(x)
    if len(curr_set) > best_size:
        best_size = len(curr_set)
        best_set = set(curr_set)
    
    # ILS
    while time.time() < deadline:
        # Perturbation: Remove a few random elements from current best
        # and try to refill greedily
        curr = list(best_set)
        rng.shuffle(curr)
        
        # Remove 2-5 elements
        num_remove = rng.randint(2, 5)
        if len(curr) > num_remove:
            to_remove = set(curr[:num_remove])
            curr = [x for x in curr if x not in to_remove]
        else:
            curr = []
        
        # Refill: Try adding all remaining numbers in random order (or sorted descending)
        # Sorted descending is generally better for packing large numbers
        candidates = [x for x in range(192, 0, -1) if x not in curr]
        rng.shuffle(candidates) # Randomize to explore different local minima
        
        # Greedy add
        for x in candidates:
            if can_add(curr, x):
                curr.append(x)
        
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = set(curr)
            # Update deadline if we want to run longer? No, fixed budget.
            
        # Local Search: Try swapping one element for another to fix conflicts or add new ones
        # Actually, can_add is strict. If we can't add, we are stuck.
        # Swap: Remove r, try to add a.
        # We can do a quick local search on the current `curr`
        improved = True
        while improved and time.time() < deadline:
            improved = False
            curr_list = list(curr)
            rng.shuffle(curr_list)
            
            # Try removing one and adding one
            for r in curr_list[:10]: # Try a few removals
                # Find a candidate to add
                for a in range(192, 0, -1):
                    if a in curr:
                        continue
                    # Check if (curr - {r} + {a}) is valid
                    # This requires checking validity of the new set.
                    # can_add is for adding to an existing valid set.
                    # (curr - {r}) is valid. We can check if we can add 'a' to (curr - {r}).
                    # But wait, 'r' might have been blocking 'a'? No, 'a' wasn't in curr.
                    # So if we remove r, the set becomes smaller. It's "easier" to add elements.
                    # So if 'a' was not addable to curr, it might be addable to curr-{r}.
                    
                    # Efficient check:
                    # Create temp list
                    temp_list = [x for x in curr if x != r]
                    if can_add(temp_list, a):
                        curr = set(curr)
                        curr.remove(r)
                        curr.add(a)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                if len(curr) > best_size:
                    best_size = len(curr)
                    best_set = set(curr)

    return list(best_set)
