import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    def time_left():
        return deadline - time.time()

    # Precompute conflict masks for fast validity checking
    # conflict[u][v] = bitmask of all w such that {u,v,w} forms a forbidden triple
    # A forbidden triple: distinct a,b,c where a+b, a+c, b+c are all in S
    # For a given pair (u,v) in S, the "forbidden" w's are those where
    # adding w would create a triple with u and v.
    
    # Actually, let me think about this more carefully.
    # A set S is valid if there are no three DISTINCT a,b,c in S such that
    # a+b, a+c, b+c are ALL in S.
    # 
    # When we add a new element x to S, we need to check:
    # 1. Triples where x is one of {a,b,c}: for each pair (a,b) in S, check if
    #    {a,b,x} is a forbidden triple. This means a+b, a+x, b+x must all be in S∪{x}.
    #    Since a+x and b+x > x (for positive a,b), they must be in S.
    #    And a+b must be in S or a+b == x.
    # 2. Triples where x is one of the sums: for each pair (a,b) in S with a+b == x,
    #    check if there exists c in S such that a+c and b+c are in S.
    
    # Let me use a different approach: precompute for each pair (u,v) what values of w
    # would make {u,v,w} a forbidden triple.
    
    # For distinct u, v, w, {u,v,w} is forbidden iff:
    # u+v, u+w, v+w are all in S.
    # So for a fixed pair (u,v), w is "conflicting" if u+w and v+w are both in S
    # and u+v is in S (or u+v == w).
    
    # This is still dependent on S, so we can't fully precompute it.
    # But we can precompute the structure.
    
    # Let me use a smarter approach with numpy for speed.
    
    import numpy as np
    
    N = 128
    
    # Precompute all pairs and their sums
    # pair_sums[i][j] = i + j for i < j
    # We'll use bitmasks for efficiency
    
    # For the greedy construction, let's use a fast approach:
    # Represent S as a boolean array and use numpy operations
    
    best_mask = 0
    best_size = 0
    rng = random.Random(42)
    
    def mask_to_set(mask):
        return [i for i in range(1, N + 1) if (mask >> i) & 1]
    
    def can_add(mask, x):
        """Check if x can be added to set represented by mask."""
        # Check triples where x is a member: (a, b, x)
        # Need a+b in S or a+b==x, and a+x in S, and b+x in S
        # a+x <= N => a <= N-x
        # b+x <= N => b <= N-x
        max_ab = N - x
        if max_ab >= 1:
            # Iterate over a in S where a <= max_ab and a+x in S
            for a in range(1, max_ab + 1):
                if not ((mask >> a) & 1):
                    continue
                xa = x + a
                if xa > N or not ((mask >> xa) & 1):
                    continue
                # Now check b > a, b <= max_ab, b in S, b+x in S, and (a+b in S or a+b==x)
                for b in range(a + 1, max_ab + 1):
                    if not ((mask >> b) & 1):
                        continue
                    xb = x + b
                    if xb > N or not ((mask >> xb) & 1):
                        continue
                    ab = a + b
                    if ab == x:
                        return False
                    if ab <= N and ((mask >> ab) & 1):
                        return False
        
        # Check triples where x is a sum: (a, b, c) with a+b == x
        for a in range(1, x // 2 + 1):
            b = x - a
            if b <= a or b > N:
                continue
            if not ((mask >> a) & 1) or not ((mask >> b) & 1):
                continue
            # Check if exists c != a, c != b in S with a+c in S and b+c in S
            max_c = N - b
            for c in range(1, max_c + 1):
                if c == a or c == b:
                    continue
                if not ((mask >> c) & 1):
                    continue
                ac = a + c
                bc = b + c
                if ac <= N and bc <= N and ((mask >> ac) & 1) and ((mask >> bc) & 1):
                    return False
        
        return True
    
    def is_valid(mask):
        """Verify the set is valid."""
        elems = mask_to_set(mask)
        n = len(elems)
        for i in range(n):
            a = elems[i]
            for j in range(i + 1, n):
                b = elems[j]
                ab = a + b
                if ab > N:
                    continue
                if not ((mask >> ab) & 1):
                    continue
                for k in range(j + 1, n):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac > N or bc > N:
                        continue
                    if ((mask >> ac) & 1) and ((mask >> bc) & 1):
                        return False
        return True
    
    # Strategy: Use randomized greedy with multiple restarts and local search
    # Key insight: starting with large numbers tends to work well since sums exceed 128
    
    def greedy_construction(base_start, order_seed=None):
        """Greedy construction starting from a base set."""
        local_rng = random.Random(order_seed) if order_seed is not None else rng
        mask = 0
        
        # Start with upper portion
        for i in range(base_start, N + 1):
            mask |= (1 << i)
        
        # Try to add lower elements in random order
        lower = list(range(1, base_start))
        local_rng.shuffle(lower)
        
        for x in lower:
            if time_left() < 10:
                break
            if not ((mask >> x) & 1) and can_add(mask, x):
                mask |= (1 << x)
        
        return mask
    
    def local_search(mask, max_iter=200000):
        """Local search: try swaps to improve."""
        current_mask = mask
        current_size = bin(current_mask).count('1')
        no_improve = 0
        
        for _ in range(max_iter):
            if time_left() < 5:
                break
            
            made_progress = False
            
            # Try to add an element
            for x in range(1, N + 1):
                if (current_mask >> x) & 1:
                    continue
                if can_add(current_mask, x):
                    current_mask |= (1 << x)
                    current_size += 1
                    made_progress = True
                    break
            
            if made_progress:
                no_improve = 0
                continue
            
            # Try swaps
            elems = mask_to_set(current_mask)
            found = False
            for r in elems:
                if time_left() < 5:
                    break
                without = current_mask & ~(1 << r)
                best_a = -1
                for a in range(1, N + 1):
                    if (without >> a) & 1:
                        continue
                    if can_add(without, a):
                        test_mask = without | (1 << a)
                        # Check if we can add more
                        can_grow = False
                        for x2 in range(1, N + 1):
                            if (test_mask >> x2) & 1:
                                continue
                            if can_add(test_mask, x2):
                                can_grow = True
                                break
                        if can_grow:
                            best_a = a
                            break
                        if a > r and (best_a == -1 or a > best_a):
                            best_a = a
                
                if best_a != -1:
                    current_mask = without | (1 << best_a)
                    current_size = bin(current_mask).count('1')
                    made_progress = True
                    found = True
                    break
            
            if made_progress:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 5:
                    break
        
        return current_mask, current_size
    
    # Run multiple strategies
    # Strategy 1: Upper half base
    cur_mask = greedy_construction(65, seed=1)
    cur_size = bin(cur_mask).count('1')
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    if time_left() > 20:
        cur_mask, cur_size = local_search(cur_mask)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
    
    # Strategy 2: Random restarts with different base starts
    for trial in range(50):
        if time_left() < 15:
            break
        base_start = random.Random(trial * 7 + 3).randint(50, 70)
        cur_mask = greedy_construction(base_start, seed=trial)
        cur_size = bin(cur_mask).count('1')
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
        
        if time_left() > 15:
            cur_mask, cur_size = local_search(cur_mask, max_iter=100000)
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
    
    # Strategy 3: Odds and evens
    for parity in [0, 1]:
        if time_left() < 10:
            break
        cur_mask = 0
        for i in range(1, N + 1):
            if i % 2 == parity:
                cur_mask |= (1 << i)
        cur_size = bin(cur_mask).count('1')
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
    
    # Final polish
    if time_left() > 5:
        for x in range(1, N + 1):
            if time_left() < 3:
                break
            if (best_mask >> x) & 1:
                continue
            if can_add(best_mask, x):
                best_mask |= (1 << x)
                best_size += 1
    
    result = mask_to_set(best_mask)
    
    # Verify
    if not is_valid(best_mask):
        result = list(range(65, 129))
    
    return [int(x) for x in result]
