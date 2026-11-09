<<<<<<< SEARCH
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin out of 60s
    
    N = 256
    MAX_VAL = N
    
    # Fast AP check: given a boolean array `in_set` and candidate x,
    # check if adding x creates a 3-AP. O(N) worst case but with early exit.
    def is_valid_addition_fast(in_set, x):
        """Check if x can be added to the set represented by in_set without creating a 3-AP."""
        # Case 1: x is the first term (smallest). Need b, c in set with 2b = x+c, x < b < c.
        for b in range(x + 1, MAX_VAL + 1):
            if in_set[b]:
                c = 2 * b - x
                if c <= MAX_VAL and in_set[c]:
                    return False
        
        # Case 2: x is the middle term. Need a, c in set with 2x = a+c, a < x < c.
        for a in range(1, x):
            if in_set[a]:
                c = 2 * x - a
                if c <= MAX_VAL and in_set[c]:
                    return False
        
        # Case 3: x is the last term (largest). Need a, b in set with 2b = a+x, a < b < x.
        for b in range(1, x):
            if in_set[b]:
                a = 2 * b - x
                if a >= 1 and in_set[a]:
                    return False
        
        return True
    
    # Initial construction: Greedy from smallest to largest
    in_set = [False] * (MAX_VAL + 1)
    S = set()
    for x in range(1, MAX_VAL + 1):
        if is_valid_addition_fast(in_set, x):
            in_set[x] = True
            S.add(x)
=======
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin out of 60s
    
    N = 256
    MAX_VAL = N
    
    # Fast AP check: given a boolean array `in_set` and candidate x,
    # check if adding x creates a 3-AP. O(N) worst case but with early exit.
    def is_valid_addition_fast(in_set, x):
        """Check if x can be added to the set represented by in_set without creating a 3-AP."""
        # Case 1: x is the first term (smallest). Need b, c in set with 2b = x+c, x < b < c.
        for b in range(x + 1, MAX_VAL + 1):
            if in_set[b]:
                c = 2 * b - x
                if c <= MAX_VAL and in_set[c]:
                    return False
        
        # Case 2: x is the middle term. Need a, c in set with 2x = a+c, a < x < c.
        for a in range(1, x):
            if in_set[a]:
                c = 2 * x - a
                if c <= MAX_VAL and in_set[c]:
                    return False
        
        # Case 3: x is the last term (largest). Need a, b in set with 2b = a+x, a < b < x.
        for b in range(1, x):
            if in_set[b]:
                a = 2 * b - x
                if a >= 1 and in_set[a]:
                    return False
        
        return True
    
    # Initial construction: Base-11 digit-free set (no zero digits in base 11)
    # This gives a guaranteed 3-AP-free set with good reciprocal weight
    def has_zero_digit_base11(n):
        """Check if n has a zero digit when written in base 11."""
        while n > 0:
            if n % 11 == 0:
                return True
            n //= 11
        return False
    
    in_set = [False] * (MAX_VAL + 1)
    S = set()
    for x in range(1, MAX_VAL + 1):
        if not has_zero_digit_base11(x):
            in_set[x] = True
            S.add(x)
>>>>>>> REPLACE

