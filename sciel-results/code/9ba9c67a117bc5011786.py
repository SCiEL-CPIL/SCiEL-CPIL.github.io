import time

def construct():
    n = 128
    start_time = time.time()
    deadline = 80.0  # 80s limit, 10s margin
    
    best_set = set()
    
    def is_valid_addition(S, x):
        """
        Check if adding x to S creates a forbidden triple.
        A forbidden triple a,b,c in S U {x} must involve x as one of the sums.
        So we check if there exist distinct a,b,c in S such that:
        1. a+b == x and a+c in S and b+c in S
        2. a+c == x and a+b in S and b+c in S
        3. b+c == x and a+b in S and a+c in S
        """
        # Optimization: iterate through elements in S to find pairs summing to x
        # Or just iterate through all pairs in S?
        # |S| is small (~80). Pairs ~3200.
        # For each pair (a,b) with a<b, let s = a+b.
        # If s == x, we need to check if there exists c in S \ {a,b} such that a+c in S and b+c in S.
        
        # Let's just check the 3 conditions directly by iterating c?
        # No, iterating pairs is better.
        
        # Actually, we can just check:
        # For all a in S:
        #   For all b in S with b > a:
        #     s = a + b
        #     if s == x:
        #        # Check if there is c in S \ {a,b} such that a+c in S and b+c in S
        #        for c in S:
        #           if c != a and c != b:
        #              if a+c in S and b+c in S:
        #                 return False
        #     elif s in S:
        #        # This is a potential triple sum in S.
        #        # If we add x, does it complete a triple?
        #        # The triple would be a,b,c where one sum is x.
        #        # We already handled s==x.
        #        # What if x is a+c? Then c = x-a.
        #        # If c in S and c != a and c != b:
        #        #    Check if b+c in S. (And a+b in S is already true)
        #        # What if x is b+c? Then c = x-b.
        #        # If c in S and c != a and c != b:
        #        #    Check if a+c in S.
        
        pass

    # Better logic for is_valid_addition:
    # We need to ensure NO distinct a,b,c in S U {x} have all 3 sums in S U {x}.
    # Since S is valid, only triples involving x as a SUM are at risk.
    # The triple elements a,b,c must be in S.
    # The sums are a+b, a+c, b+c.
    # One of these sums must be x.
    
    def can_add(S, x):
        # Case 1: x = a+b. Need a+c in S, b+c in S for some c in S \ {a,b}.
        # Case 2: x = a+c. Need a+b in S, b+c in S for some c in S \ {a,b}. (Symmetric to 1)
        # Case 3: x = b+c. Need a+b in S, a+c in S for some c in S \ {a,b}. (Symmetric to 1)
        
        # So we just need to check:
        # Does there exist distinct a,b,c in S such that:
        # (a+b == x and a+c in S and b+c in S) OR
        # (a+c == x and a+b in S and b+c in S) OR
        # (b+c == x and a+b in S and a+c in S)
        
        # We can iterate through pairs (a,b) in S.
        for a in S:
            for b in S:
                if a >= b: continue
                s_ab = a + b
                if s_ab == x:
                    # Check if there is c
                    for c in S:
                        if c == a or c == b: continue
                        if a + c in S and b + c in S:
                            return False
                # Check if x is a+c
                # a+c = x => c = x-a
                c = x - a
                if c in S and c != a and c != b:
                    if a + b in S and b + c in S:
                        return False
                # Check if x is b+c
                # b+c = x => c = x-b
                c = x - b
                if c in S and c != a and c != b:
                    if a + b in S and a + c in S:
                        return False
        return True

    # Greedy construction
    S = set()
    for x in range(1, n + 1):
        if can_add(S, x):
            S.add(x)
    
    # Local search / Randomized restarts to improve
    # Try to add elements that were skipped?
    # Or swap?
    # Since we want max size, and greedy is monotonic, we can try different orders.
    
    best_size = len(S)
    best_result = list(S)
    
    # Try a few randomized permutations
    import random
    for _ in range(50):
        if time.time() > deadline: break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_rand = set()
        for x in order:
            if can_add(S_rand, x):
                S_rand.add(x)
        if len(S_rand) > best_size:
            best_size = len(S_rand)
            best_result = list(S_rand)
            
    return best_result
