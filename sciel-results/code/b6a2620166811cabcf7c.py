import time
import random
import numpy as np

def construct():
    n = 128
    start_time = time.time()
    deadline = 85.0
    
    # Bitmask representation: bit i represents whether i is in the set (1-indexed)
    # We use bits 1..n (bit 0 unused)
    
    def bitmask_to_set(mask):
        return [i for i in range(1, n + 1) if (mask >> i) & 1]
    
    def set_to_bitmask(S):
        mask = 0
        for x in S:
            mask |= (1 << x)
        return mask
    
    # Precompute for each pair (a,b) with a < b, the "conflict" condition
    # A triple (a,b,c) with a<b<c is forbidden if a+b, a+c, b+c are all in S.
    # When we add x, we need to check if x creates a forbidden triple.
    
    # For bit-parallel: given current set S (as bitmask), check if adding x is valid.
    # A forbidden triple involving x:
    #   Case 1: x is one of a,b,c. WLOG x=a. Need b,c in S with x+b in S, x+c in S, b+c in S.
    #   Case 2: x is one of the sums. WLOG x=a+b. Need a,b in S, a!=b, and some c in S\{a,b} with a+c in S and b+c in S.
    
    # Precompute sum masks: for each value v, sum_mask[v] = bitmask of all pairs (a,b), a<b, a+b=v
    # Actually, let's think differently.
    
    # For efficiency, precompute:
    # pair_sum_mask[v]: for each v in 2..2n, the set of all a in 1..n-1 such that a < v-a (i.e., a < v/2)
    #   This gives us the "smaller" elements of pairs summing to v.
    
    # Let me use a different approach: precompute for each x, the list of "conflict patterns"
    
    # Actually, the most efficient bit-parallel approach:
    # For adding x to S:
    #   Case 1 (x is a vertex): For each b in S where x+b <= n and x+b in S:
    #     For each c in S where c > b, x+c <= n, x+c in S:
    #       Check if b+c in S or b+c == x
    #   This is still O(n^2) in worst case.
    
    # Better: use bitmasks for the sums.
    # Let S be the bitmask. For a given x:
    #   Case 1: x+b in S means b is in S AND (S >> x) & (1 << b). 
    #   So the set of b such that x+b in S is: B_x = S & (S >> x) ... no.
    #   x+b in S means bit (x+b) is set in S. So b is valid if bit b is set in S AND bit (x+b) is set in S.
    #   B_x = S & (S >> x) gives us bits b where both b and x+b are in S. But S >> x shifts right,
    #   so bit b in (S >> x) means bit (b+x) in S. Yes! B_x = S & (S >> x).
    
    #   Then for each b in B_x, we need c > b in S with x+c in S and (b+c in S or b+c==x).
    #   C_x(b) = {c > b : c in S, x+c in S, b+c in S or b+c == x}
    #   = {c > b : c in S, x+c in S} ∩ {c : b+c in S or b+c == x}
    #   The first part: (S & (S >> x)) >> (b+1) ... this gets complicated.
    
    # Let me try a hybrid: use bitmask for quick filtering, then verify.
    
    def can_add_bitmask(S_mask, x):
        # S_mask is the bitmask of current set, x is not in S_mask
        # Returns True if adding x keeps validity
        
        # Case 1: x is one of a, b, c. WLOG a = x.
        # Need b, c in S, b < c, such that x+b in S, x+c in S, and (b+c in S or b+c == x)
        # B_x = {b in S : x+b in S} = S_mask & (S_mask >> x)  -- bits b where b and x+b both in S
        B_x = S_mask & (S_mask >> x)
        
        # For each b in B_x (b < x potentially, or any b), check if there exists c > b in S
        # with x+c in S and (b+c in S or b+c == x)
        # 
        # Iterate over set bits in B_x:
        b = 0
        while B_x:
            # Get lowest set bit
            lb = B_x & (-B_x)
            b = lb.bit_length() - 1
            B_x ^= lb
            if b < 1:  # skip bit 0
                continue
            # Now need c > b, c in S, x+c in S, (b+c in S or b+c == x)
            # C_candidates = {c > b : c in S and x+c in S} = (S_mask & (S_mask >> x)) >> (b+1)
            # But we also need b+c in S or b+c == x
            # For each such c, check if (S_mask >> (b+c)) & 1 or b+c == x
            
            C_cand = (S_mask & (S_mask >> x)) >> (b + 1)
            # Now iterate over c in C_cand:
            c_mask = C_cand
            while c_mask:
                lb_c = c_mask & (-c_mask)
                c = lb_c.bit_length() - 1
                c_mask ^= lb_c
                if c < 1:
                    continue
                bc = b + c
                if bc == x:
                    return False
                if bc <= n and (S_mask >> bc) & 1:
                    return False
        
        # Case 2: x is one of the sums. WLOG a + b = x, a < b.
        # Need a, b in S, a != b, a + b = x.
        # Then need c in S \ {a, b} such that a+c in S and b+c in S.
        # a ranges from 1 to (x-1)//2, b = x - a
        a_max = (x - 1) // 2
        for a in range(1, a_max + 1):
            b = x - a
            if b < 1 or b > n or b <= a:
                continue
            # Check a and b are both in S
            if not ((S_mask >> a) & 1 and (S_mask >> b) & 1):
                continue
            # Need c in S \ {a, b} with a+c in S and b+c in S
            # c > max(a,b) potentially, but c can be anything in S except a,b
            # a+c in S and b+c in S
            # C_cand2 = {c : c in S, c != a, c != b, a+c in S, b+c in S}
            # = S_mask & (S_mask >> a) & (S_mask >> b) with bits a and b cleared
            C2 = S_mask & (S_mask >> a) & (S_mask >> b)
            C2 &= ~(1 << a)
            C2 &= ~(1 << b)
            if C2:
                # There exists such c
                return False
        
        return True
    
    # Initial seed: top half
    S0 = set(range(n // 2 + 1, n + 1))
    best_mask = set_to_bitmask(S0)
    best_size = len(S0)
    
    # Greedy construction with random orders
    candidates = list(range(1, n // 2 + 1))
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        S_mask = set_to_bitmask(S0)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if not (S_mask >> x) & 1:
                if can_add_bitmask(S_mask, x):
                    S_mask |= (1 << x)
        
        sz = S_mask.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = S_mask
    
    # Local search with swaps (the key improvement from measured results)
    S_mask = best_mask
    current_size = best_size
    no_improve_count = 0
    
    while time.time() < deadline:
        if current_size == 0:
            break
        
        # Pick a random element to remove
        S_list = [i for i in range(1, n + 1) if (S_mask >> i) & 1]
        if not S_list:
            break
        y = random.choice(S_list)
        S_mask ^= (1 << y)
        
        # Try to add all possible elements
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if not (S_mask >> x) & 1:
                if can_add_bitmask(S_mask, x):
                    S_mask |= (1 << x)
        
        new_size = S_mask.bit_count()
        if new_size > best_size:
            best_size = new_size
            best_mask = S_mask
            no_improve_count = 0
        elif new_size > current_size:
            # Improved but not best -- accept (simulated annealing style)
            current_size = new_size
            no_improve_count += 1
        else:
            # Revert
            S_mask = best_mask
            current_size = best_size
            no_improve_count += 1
        
        # Occasionally try a swap: remove two, add one or two
        if no_improve_count % 3 == 0 and time.time() < deadline:
            S_mask = best_mask
            S_list = [i for i in range(1, n + 1) if (S_mask >> i) & 1]
            if len(S_list) >= 2:
                y1, y2 = random.sample(S_list, 2)
                S_mask ^= (1 << y1)
                S_mask ^= (1 << y2)
                for x in range(1, n + 1):
                    if time.time() > deadline:
                        break
                    if not (S_mask >> x) & 1:
                        if can_add_bitmask(S_mask, x):
                            S_mask |= (1 << x)
                new_size = S_mask.bit_count()
                if new_size > best_size:
                    best_size = new_size
                    best_mask = S_mask
                    no_improve_count = 0
    
    result = [i for i in range(1, n + 1) if (best_mask >> i) & 1]
    return result
