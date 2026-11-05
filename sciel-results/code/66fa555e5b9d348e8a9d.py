import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    
    # Base set: {171, ..., 256}
    # This set is internally valid because for any a < b <= c in this set,
    # a >= 171, so 3a >= 513 > 512 >= b+c.
    # And b+c > 2a because b > a.
    # So no multiple of a falls in (2a, 512].
    S = set(range(171, n + 1))
    
    # We want to add elements from 170 down to 1.
    # For a candidate x, it can be added if for all b, c in S with b > x,
    # x does not divide b + c.
    # Since we iterate downwards, all elements in S are > x.
    # So we just check all pairs in S.
    
    # To speed up, we can precompute sums or use bitsets?
    # |S| is at most 256. Pairs ~ 32k.
    # 170 candidates * 32k checks = 5.4M ops. Fast.
    
    for x in range(170, 0, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Check if x is compatible with current S
        valid = True
        S_list = sorted(S)
        m = len(S_list)
        
        # Optimization: if x is large, few multiples.
        # Just check all pairs.
        for i in range(m):
            b = S_list[i]
            # b > x is guaranteed since we go down and S contains > x
            # Check b+b
            if (b + b) % x == 0:
                valid = False
                break
            for j in range(i + 1, m):
                c = S_list[j]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.add(x)
            
    # The set S is now built.
    # We should verify the entire set to be safe.
    # The logic above ensures that when x was added, it didn't conflict with existing elements.
    # And existing elements were already valid among themselves (by induction, base case S_large is valid).
    # So S should be valid.
    
    return sorted(S)
