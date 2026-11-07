import time

def construct():
    start_time = time.time()
    deadline = 90.0
    margin = 5.0
    
    # Initial safe set: top half
    S = set(range(65, 129))
    
    # Try to add elements from 1 to 64
    # Order: ascending
    for x in range(1, 65):
        if time.time() > deadline - margin:
            break
            
        # Check if adding x creates a forbidden triple
        valid = True
        # We need to check if there exist a, b in S such that:
        # x+a in S_new, x+b in S_new, a+b in S_new
        # Since x is new, x+a > x, so x+a must be in S (old)
        # x+b must be in S (old)
        # a+b must be in S (old) or a+b == x
        
        # Optimization:
        # x+a in S => a in S-x
        # x+b in S => b in S-x
        # So a and b must be in S and also in S-x.
        # Let candidates = S & (S - x)  -- wait, S-x is {y-x | y in S}
        # We need a, b in S such that a+x in S and b+x in S.
        # So a is in S and a+x is in S.
        # Let A = {a in S | a+x in S}
        # We need distinct a, b in A such that a+b in S or a+b == x.
        
        A = []
        for a in S:
            if a + x in S:
                A.append(a)
        
        # Now check pairs in A
        n = len(A)
        if n < 2:
            # Can't form a pair
            S.add(x)
            continue
            
        # Check all pairs in A
        # If n is small, O(n^2) is fine.
        # Max size of S is 128. A is subset of S.
        for i in range(n):
            a = A[i]
            for j in range(i+1, n):
                b = A[j]
                # a, b distinct
                # Check a+b
                if a + b in S or a + b == x:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.add(x)
            
    # Convert to list of ints
    return sorted(list(S))
