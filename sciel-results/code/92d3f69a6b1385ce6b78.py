"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 256
    
    def is_valid_addition(x, S):
        """
        Check if adding x to set S (list of ints) maintains the property:
        For every a < b <= c in S, a does NOT divide b+c.
        
        We only need to check violations involving x, since S is already valid.
        Violations can occur where x is:
        1. The 'a' in a < b <= c: x < b <= c, x | (b+c)
        2. The 'b' in a < b <= c: a < x <= c, a | (x+c)
        3. The 'c' in a < b <= c: a < b <= x, a | (b+x)
        """
        if not S:
            return True
        
        # Optimization: If x is the largest element, it can only be 'c' or 'b'?
        # No, if x is largest, it can be 'c' (a<b<=x) or 'b' (a<x<=c is impossible if x is max, unless c=x, so a<x=x).
        # Wait, if x is max, then:
        # 1. x as 'a': impossible since no b > x.
        # 2. x as 'b': a < x <= c. Since x is max, c must be x. So a < x <= x. Check a | (x+x) = 2x.
        # 3. x as 'c': a < b <= x. Check a | (b+x).
        
        # General case:
        # We can optimize by pre-sorting S, but S is small (<256).
        # Let's just iterate.
        
        # Case 1: x is 'a'
        # Need b, c in S such that x < b <= c and (b+c) % x == 0
        # To speed this up, we can collect all elements > x.
        # S_gt_x = [y for y in S if y > x]
        # For each pair (b, c) in S_gt_x with b <= c:
        #   if (b+c) % x == 0: return False
        
        # Case 2: x is 'b'
        # Need a, c in S such that a < x <= c and (x+c) % a == 0
        # S_lt_x = [y for y in S if y < x]
        # S_ge_x = [y for y in S if y >= x]
        # For each a in S_lt_x:
        #   For each c in S_ge_x:
        #     if (x+c) % a == 0: return False
        
        # Case 3: x is 'c'
        # Need a, b in S such that a < b <= x and (b+x) % a == 0
        # S_lt_x = [y for y in S if y < x]
        # For each a in S_lt_x:
        #   For each b in S_lt_x:
        #     if a < b <= x and (b+x) % a == 0: return False
        # Note: b can be equal to x? No, b <= x. If b=x, then a < x.
        # Wait, if b=x, then a < x. (x+x)%a == 0?
        # The condition is a < b <= c. If x is c, then b <= x.
        # If b=x, then a < x.
        
        # Let's implement a unified check to avoid code duplication and logic errors.
        # Since n=256, |S| <= 256.
        # Worst case |S|^3 is too slow if done naively for every insertion?
        # |S| grows to ~150. 150^3 = 3.3M. 256 insertions * 3.3M = 850M ops. Too slow in Python.
        
        # We need O(|S|^2) per insertion.
        
        # Let S be a list.
        # Pre-compute sorted S?
        # Let's use sets/lists for partitioning.
        
        S_list = S
        
        # Case 1: x is 'a'
        # b, c > x
        S_gt = [y for y in S_list if y > x]
        # We need to check pairs (b, c) in S_gt with b <= c.
        # (b+c) % x == 0
        # Since b,c > x, b+c > 2x.
        # We can iterate through S_gt.
        # To make it O(|S_gt|^2), we just double loop.
        # But we can optimize: for a fixed b, c must be in S_gt, c >= b, and (b+c) % x == 0.
        # This is equivalent to c % x == (-b) % x.
        # So for each b in S_gt, we check if there exists c in S_gt, c >= b, such that c % x == (-b) % x.
        # We can precompute counts or just check.
        
        # Actually, simple double loop might be fast enough if S_gt is small.
        # But let's try to be smart.
        
        # Case 2: x is 'b'
        # a < x, c >= x
        S_lt = [y for y in S_list if y < x]
        S_ge = [y for y in S_list if y >= x]
        # For each a in S_lt:
        #   For each c in S_ge:
        #     if (x+c) % a == 0: return False
        
        # Case 3: x is 'c'
        # a < b <= x
        # a, b in S_lt? No, b can be x? No, b is in S, x is not in S yet.
        # So b < x.
        # So a, b in S_lt.
        # For each a in S_lt:
        #   For each b in S_lt:
        #     if a < b and (b+x) % a == 0: return False
        
        # Implementation:
        
        # Case 1: x is 'a'
        if S_gt:
            # Optimize: group by modulo x
            # We need pairs (b, c) with b <= c and (b+c) % x == 0.
            # This means c % x == (-b) % x.
            # Let's just double loop for simplicity first, n=256 is small.
            # If S_gt is size 100, 100*100 = 10k ops.
            for i in range(len(S_gt)):
                b = S_gt[i]
                # We can break early if b + min(S_gt) > max_possible_multiple? No.
                for j in range(i, len(S_gt)):
                    c = S_gt[j]
                    if (b + c) % x == 0:
                        return False
        
        # Case 2: x is 'b'
        if S_lt and S_ge:
            for a in S_lt:
                # We need c in S_ge such that (x+c) % a == 0
                # c % a == (-x) % a
                target_rem = (-x) % a
                for c in S_ge:
                    if c % a == target_rem:
                        return False
        
        # Case 3: x is 'c'
        if S_lt:
            # a, b in S_lt, a < b
            # (b+x) % a == 0
            # b % a == (-x) % a
            for i in range(len(S_lt)):
                a = S_lt[i]
                target_rem = (-x) % a
                # b must be > a
                # We can iterate through S_lt again
                for j in range(len(S_lt)):
                    b = S_lt[j]
                    if b > a and b % a == target_rem:
                        return False
        
        return True

    def is_valid_set(S):
        """Full validation for safety."""
        S_sorted = sorted(S)
        n = len(S_sorted)
        for i in range(n):
            a = S_sorted[i]
            for j in range(i+1, n):
                b = S_sorted[j]
                for k in range(j, n):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Strategy:
    # 1. Greedy construction from high to low (safe start).
    # 2. Local search / Simulated Annealing to improve.
    
    # Phase 1: Greedy
    S = []
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_addition(x, S):
            S.append(x)
    
    best_set = S[:]
    best_size = len(S)
    
    # Phase 2: Local Search (Simulated Annealing style)
    # Moves:
    # 1. Add an element not in S.
    # 2. Remove an element from S.
    # 3. Swap: remove one, add another.
    # Objective: Maximize size.
    # Since we want to maximize size, we prefer moves that increase size.
    # If size is same, we might accept to escape local optima?
    # But size is the only metric. So we only accept if size increases.
    # This is just hill climbing.
    # To escape local optima, we need to allow temporary size decrease?
    # But the goal is to maximize size. If we decrease size, we risk not getting back.
    # However, sometimes removing 1 allows adding 2.
    # Let's try a "remove 1, add 2" move explicitly.
    # Or just random restarts of the greedy phase with different orderings.
    
    # Given the time limit, let's do multiple greedy runs with different randomizations.
    # And a targeted "add if possible" loop.
    
    # Improvement 1: Targeted addition to best_set
    # Try to add any missing element.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        S_targeted = best_set[:]
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x in S_targeted:
                continue
            if is_valid_addition(x, S_targeted):
                S_targeted.append(x)
                changed = True
        if len(S_targeted) > best_size:
            best_size = len(S_targeted)
            best_set = S_targeted
    
    # Improvement 2: Randomized Greedy Restarts
    # Try different orders to find better packings.
    # 1. Random shuffle
    # 2. Priority by "conflict likelihood" (small numbers first? large first?)
    # We already did large first. Let's try random.
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S_try = []
        for x in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S_try):
                S_try.append(x)
        
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try[:]
            
            # If we find a new best, try to polish it with targeted addition
            S_polished = best_set[:]
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if x in S_polished:
                    continue
                if is_valid_addition(x, S_polished):
                    S_polished.append(x)
            if len(S_polished) > best_size:
                best_size = len(S_polished)
                best_set = S_polished
    
    # Improvement 3: Local Search with Swaps
    # Try to swap elements in best_set to allow adding more.
    # This is complex. Let's stick to the above for now.
    # The "remove 1, add 2" logic is implicitly covered by random restarts if the random order
    # happens to place the 2 new elements before the 1 removed element? No, greedy only adds.
    # So we need explicit swap logic if we want to explore that space.
    
    # Let's add a simple swap search.
    # Pick a random element r in best_set. Remove it.
    # Try to add any two elements not in best_set?
    # Or just try to add one element to best_set - {r}.
    # If size increases (which it won't, 1 out 1 in), we don't gain.
    # We need to add MORE than we remove.
    # So: Remove 1, Add 2.
    
    # Let's try:
    # For each r in best_set (randomly sampled):
    #   S_temp = best_set - {r}
    #   Try to add any x not in S_temp.
    #   If we can add x, S_temp becomes size same as best_set.
    #   Then try to add another y not in S_temp.
    #   If we can add y, S_temp becomes size best_size + 1.
    #   Accept.
    
    # This is O(N * |S| * |S|^2) which might be slow.
    # Let's limit this to a few iterations.
    
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Sample a few random removals
        for _ in range(5):
            if time.time() > deadline:
                break
            
            if not best_set:
                continue
                
            r = random.choice(best_set)
            S_temp = [x for x in best_set if x != r]
            
            # Try to add one element
            added_one = False
            for x in range(1, n + 1):
                if x in S_temp:
                    continue
                if is_valid_addition(x, S_temp):
                    S_temp.append(x)
                    added_one = True
                    break
            
            if not added_one:
                continue
                
            # Try to add a second element
            added_two = False
            for x in range(1, n + 1):
                if x in S_temp:
                    continue
                if is_valid_addition(x, S_temp):
                    S_temp.append(x)
                    added_two = True
                    break
            
            if added_two:
                # Size increased by 1
                if len(S_temp) > best_size:
                    best_size = len(S_temp)
                    best_set = S_temp[:]
                    # Polish
                    S_polished = best_set[:]
                    for x in range(1, n + 1):
                        if time.time() > deadline:
                            break
                        if x in S_polished:
                            continue
                        if is_valid_addition(x, S_polished):
                            S_polished.append(x)
                    if len(S_polished) > best_size:
                        best_size = len(S_polished)
                        best_set = S_polished

    return best_set
# EVOLVE-BLOCK-END
