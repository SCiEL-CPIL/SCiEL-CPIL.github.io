def construct():
    """
    Constructs a large set in 1..128 with no triple a,b,c such that
    a+b, a+c, b+c are all in the set.
    Uses a greedy approach adding numbers from 128 down to 1.
    """
    S = set()
    
    # We check candidates from 128 down to 1
    # This favors large numbers whose sums exceed 128
    for candidate in range(128, 0, -1):
        # Check if adding candidate creates a violation
        # A violation exists if there are distinct a, b, c in S U {candidate}
        # such that a+b, a+c, b+c are all in S U {candidate}
        
        # Efficient check:
        # 1. If candidate is added, it can be part of a triple.
        #    Case A: candidate, x, y where x, y in S.
        #    Sums: candidate+x, candidate+y, x+y.
        #    All must be in S U {candidate}.
        #    Note: x+y must be in S (since x,y in S, x+y != candidate usually, unless specific case).
        #    Actually x+y could be candidate? x+y=candidate.
        #    Then we need candidate+x in S U {candidate} and candidate+y in S U {candidate}.
        
        # 2. Triples not involving candidate are already safe.
        
        valid = True
        # We need to check if there exist x, y in S such that:
        # (candidate + x) in S | {candidate}
        # (candidate + y) in S | {candidate}
        # (x + y) in S | {candidate}
        
        # Optimization:
        # If x+y > 128, then x+y not in S (max 128).
        # So we only need to check pairs x,y in S with x+y <= 128.
        
        # Also, candidate+x must be <= 128 to be in S (or equal to candidate).
        # If candidate+x > 128, it's not in S.
        
        # Let's iterate through pairs in S?
        # Size of S is up to ~64. Pairs ~2000.
        # 128 candidates * 2000 pairs = 256,000 checks. Very fast.
        
        S_list = list(S)
        n = len(S_list)
        
        for i in range(n):
            for j in range(i + 1, n):
                x = S_list[i]
                y = S_list[j]
                
                sum_xy = x + y
                sum_cx = candidate + x
                sum_cy = candidate + y
                
                # Check if all three sums are in S U {candidate}
                # S is a set, so lookup is O(1)
                
                if sum_xy in S or sum_xy == candidate:
                    if sum_cx in S or sum_cx == candidate:
                        if sum_cy in S or sum_cy == candidate:
                            # Violation found
                            valid = False
                            break
            if not valid:
                break
        
        if valid:
            S.add(candidate)
            
    return list(S)
