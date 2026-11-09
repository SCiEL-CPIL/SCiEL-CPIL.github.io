import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    all_mask = (1 << n) - 1
    
    # Precompute conflict masks: for each element x, which elements y conflict with x
    # x and y conflict if adding both creates a forbidden triple
    # A triple (a,b,c) is forbidden if a+b, a+c, b+c are all in the set
    # So x conflicts with y if there exists z such that {x,y,z} forms a bad triple
    # OR if x+y is in the set and some z makes it bad
    
    # Better approach: precompute for each pair (x,y) whether they can coexist
    # Actually, let's precompute the "conflict" relationship:
    # Two elements x, y conflict if there exists z such that all of x, y, z, x+y, x+z, y+z are in domain [1..128]
    # and if we have x,y,z in our set, then x+y, x+z, y+z must NOT all be in our set.
    
    # For greedy MIS: process vertices in random order, add vertex if it doesn't conflict with any already-added vertex.
    # Conflict: x and y conflict if there exists z in [1..128] such that:
    #   x+y <= 128, x+z <= 128, y+z <= 128
    #   and if x,y,z are all selected, then x+y, x+z, y+z would need to be checked.
    # Actually the constraint is: NO three distinct a,b,c in S such that a+b, a+c, b+c are ALL in S.
    # So x and y don't directly conflict; they conflict via a third element z.
    
    # For greedy: when considering adding x, check if it creates a bad triple with any two already-selected elements.
    # This is O(|S|^2) per addition, which is too slow for 128 elements.
    
    # Alternative: use bit-parallel checking.
    # For a given set S (as bitmask), and candidate x, check validity:
    # Case 1: x is the "third vertex" c. Need a,b in S, a!=b, a!=x, b!=x, such that a+x in S, b+x in S, a+b in S.
    # Case 2: x is a sum, say x = a+b. Need c in S, c!=a, c!=b, such that a+c in S, b+c in S.
    
    # Let's precompute for each x the list of (a,b) pairs where x could be c=a+b or x could be a vertex.
    
    # Precompute: for each x in 1..128, store:
    #   sums_of_x: list of (a,b) with a<b, a+b=x, both in 1..128
    #   pairs_with_x: for case where x is a vertex, we need a,b such that a+x in range, b+x in range, a+b in range
    #   Actually for case 1 (x is c): a,b in S, a+x in S, b+x in S, a+b in S
    #   We can precompute for each x the set of (a,b) pairs where a+x<=128, b+x<=128, a+b<=128
    
    # Let me just implement a fast can_add function using bitmasks.
    
    def can_add(mask, x):
        """Check if x can be added to set represented by mask. x is 1-indexed."""
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return True
        
        # Case 1: x is the third vertex c.
        # mask_a = {a in S : a+x in S, a != x}, a in 1..n-x
        if x <= n:
            mask_a = (mask & (mask >> x)) & ((1 << (n - x)) - 1)
            if x - 1 < n - x:
                mask_a &= ~(1 << (x - 1))
            
            if mask_a & (mask_a - 1):  # at least 2 bits set
                # Check if there exist a != b in mask_a with a+b == x or a+b in S
                # Part 1: a+b == x
                # For each a in mask_a with a < x-a, check if x-a in mask_a
                # Equivalently: for a in 1..x//2, check a in mask_a and x-a in mask_a and a != x-a
                for a in range(1, x // 2 + 1):
                    b = x - a
                    if b == a:
                        continue
                    if (mask_a >> (a - 1)) & 1 and (mask_a >> (b - 1)) & 1:
                        return False
                
                # Part 2: a+b in S for some a != b in mask_a
                # For each a in mask_a, check if (mask_a \ {a}) & {b : a+b in S} != 0
                # {b : a+b in S} = (mask >> a) & ((1 << (n - a)) - 1)
                m = mask_a
                while m:
                    lb = m & (-m)
                    a = lb.bit_length()  # 1-indexed
                    # b must be in mask_a, b != a, and a+b in S
                    # a+b in S means bit (a+b-1) in mask, i.e., b in (mask >> a)
                    # b range: 1..n-a
                    b_candidates = (mask_a & ~(1 << (a - 1))) & (mask >> a) & ((1 << (n - a)) - 1)
                    if b_candidates:
                        return False
                    m ^= lb
        
        # Case 2: x = a + b for some a < b, a+b = x, a,b in S
        # Need c in S, c != a, c != b, such that a+c in S and b+c in S
        for a in range(1, x // 2 + 1):
            b = x - a
            if b <= a:
                continue
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                C_mask = (mask & (mask >> a)) & (mask & (mask >> b))
                C_mask &= ~(1 << (a - 1))
                C_mask &= ~(1 << (b - 1))
                if C_mask != 0:
                    return False
        
        return True
    
    def greedy_mis(order):
        """Greedy maximal independent set with given vertex order."""
        mask = 0
        for x in order:
            if can_add(mask, x):
                mask |= (1 << (x - 1))
        return mask
    
    def local_search(mask, time_limit):
        """Local search: try to add/remove/swap elements to improve."""
        best = mask
        best_size = bin(mask).count('1')
        deadline = time.time() + time_limit
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline:
            if no_improve > max_no_improve:
                # Reset by trying a random restart of the local search
                break
            
            improved = False
            
            # Try adding a random missing element
            missing = [x for x in range(1, n + 1) if not (mask >> (x - 1)) & 1]
            if missing:
                x = random.choice(missing)
                if can_add(mask, x):
                    mask |= (1 << (x - 1))
                    sz = bin(mask).count('1')
                    if sz > best_size:
                        best_size = sz
                        best = mask
                        no_improve = 0
                    else:
                        no_improve += 1
                    improved = True
            
            # Try swap: remove one present, add one missing
            if random.random() < 0.3:
                present = [x for x in range(1, n + 1) if (mask >> (x - 1)) & 1]
                if present and missing:
                    x_remove = random.choice(present)
                    x_add = random.choice(missing)
                    old_mask = mask
                    mask &= ~(1 << (x_remove - 1))
                    if can_add(mask, x_add):
                        mask |= (1 << (x_add - 1))
                        sz = bin(mask).count('1')
                        if sz > best_size:
                            best_size = sz
                            best = mask
                            no_improve = 0
                        else:
                            no_improve += 1
                        improved = True
                    else:
                        mask = old_mask
                        no_improve += 1
            
            # Occasionally remove an element to allow others
            if random.random() < 0.1:
                present = [x for x in range(1, n + 1) if (mask >> (x - 1)) & 1]
                if present:
                    x = random.choice(present)
                    mask &= ~(1 << (x - 1))
                    no_improve += 1
            
            if not improved:
                no_improve += 1
        
        return best
    
    best_mask = 0
    best_size = 0
    
    # Multiple restarts with different random orderings
    remaining = deadline - time.time()
    num_restarts = max(5, int(remaining / 1.0))
    
    for r in range(num_restarts):
        if time.time() >= deadline - 2:
            break
        
        # Mixed ordering: sometimes random, sometimes sorted, sometimes by "degree"
        mode = random.random()
        if mode < 0.4:
            # Random
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif mode < 0.7:
            # Sorted ascending (small numbers first - they have more potential sums)
            order = list(range(1, n + 1))
        else:
            # Sorted descending
            order = list(range(n, 0, -1))
        
        # Greedy MIS
        mask = greedy_mis(order)
        sz = bin(mask).count('1')
        
        # Local search with remaining time budget
        remaining_restarts = max(1, num_restarts - r)
        ls_time = min(2.0, (deadline - time.time()) / remaining_restarts)
        if ls_time > 0.1:
            mask = local_search(mask, ls_time)
            sz = bin(mask).count('1')
        
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Final greedy pass on best
    for x in range(1, n + 1):
        if not (best_mask >> (x - 1)) & 1:
            if can_add(best_mask, x):
                best_mask |= (1 << (x - 1))
    
    # Convert to list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    # Final verification
    result_set = set(result)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            for k in range(j + 1, len(result)):
                a, b, c = result[i], result[j], result[k]
                s1, s2, s3 = a + b, a + c, b + c
                if s1 in result_set and s2 in result_set and s3 in result_set:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return list(range(65, 129))
    
    return result
