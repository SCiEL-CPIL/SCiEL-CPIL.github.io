def construct():
    import time
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 256
    S = set()
    
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        
        can_add = True
        
        # Pre-filter or just check
        # Case 1: x is a
        # x < b <= c, b,c in S
        # Check if x divides b+c
        # b must be > x
        # c must be >= b
        
        # Case 2: x is b
        # a < x <= c, a,c in S
        # Check if a divides x+c
        
        # Case 3: x is c
        # a < b <= x, a,b in S
        # Check if a divides b+x
        
        # To optimize, we can break early if found.
        
        # We can iterate through S once and store elements > x, < x, etc.
        # But S is a set. Let's just loop.
        
        # Optimization: if x is large, few elements in S are > x.
        # If x is small, many elements in S are > x.
        
        # Let's just do the triple checks.
        
        # Case 1: x = a
        for b in S:
            if b <= x:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            # Case 2: x = b
            for a in S:
                if a >= x:
                    continue
                for c in S:
                    if c < x:
                        continue
                    if (x + c) % a == 0:
                        can_add = False
                        break
                if not can_add:
                    break
        
        if can_add:
            # Case 3: x = c
            for a in S:
                if a >= x:
                    continue
                for b in S:
                    if b <= a or b > x:
                        continue
                    if (b + x) % a == 0:
                        can_add = False
                        break
                if not can_add:
                    break
        
        if can_add:
            S.add(x)
            
    return sorted(S)
