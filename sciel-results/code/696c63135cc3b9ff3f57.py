import time

def construct():
    n = 128
    start_time = time.time()
    deadline = start_time + 50

    # Precompute d lists
    d_4s = [[] for _ in range(n + 1)]
    d_3s = [[] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        max_d_4 = (i - 1) // 3
        for d in range(1, max_d_4 + 1):
            d_4s[i].append(d)
            
        min_d_3 = (i - 1) // 3 + 1
        max_d_3 = (i - 1) // 2
        for d in range(min_d_3, max_d_3 + 1):
            d_3s[i].append(d)

    def is_valid(i, c, colors):
        # Check 3-APs (actually part of 4-AP logic, but let's stick to the 4-AP constraint directly)
        # The constraint is on 4-APs.
        # A 4-AP is (i-3d, i-2d, i-d, i).
        # We need at least 3 distinct colors in these 4 positions.
        
        for d in d_4s[i]:
            c1 = colors[i - 3 * d]
            c2 = colors[i - 2 * d]
            c3 = colors[i - d]
            c4 = c
            
            # Count distinct colors
            # If len({c1,c2,c3,c4}) < 3, invalid.
            # This is equivalent to:
            # 1. All 4 same: c1==c2==c3==c4
            # 2. 3 same, 1 diff: e.g. c1==c2==c3 != c4.
            #    In this case, the set is {c1, c4}, size 2. Invalid.
            # 3. 2 same, 2 same: c1==c2, c3==c4, c1!=c3. Set {c1, c3}, size 2. Invalid.
            # 4. 2 same, 1, 1: c1==c2, c3!=c4, c3!=c1. Set {c1, c3, c4}, size 3. Valid.
            
            # So, invalid if:
            # - c1==c2==c3==c
            # - c1==c2==c3 != c
            # - c1==c2 != c3==c
            # - c1==c3==c != c2 (permutation of above)
            # Basically, invalid if the number of distinct colors is 1 or 2.
            
            # Quick check:
            if c1 == c2 and c2 == c3:
                # c1=c2=c3. If c is same, size 1. If c diff, size 2. Both invalid.
                return False
            
            if c1 == c2:
                # c1=c2. 
                # If c3==c, set is {c1, c3}. Size 2. Invalid.
                if c3 == c:
                    return False
                # If c3 != c, set is {c1, c3, c}. 
                # If c1==c3, then c1=c2=c3, handled above.
                # So c1, c3, c are distinct? 
                # c1 != c3 (else handled). 
                # If c == c1, set {c1, c3}. Size 2. Invalid.
                if c == c1:
                    return False
                # If c == c3, handled above.
                # So if we reach here, c is distinct from c1 and c3.
                # Set is {c1, c3, c}. Size 3. Valid.
            
            elif c2 == c3:
                # c2=c3.
                # If c1==c, set {c1, c2}. Size 2. Invalid.
                if c1 == c:
                    return False
                # If c1 != c, set {c1, c2, c}.
                # If c1 == c2, handled above.
                # If c == c2, set {c1, c2}. Size 2. Invalid.
                if c == c2:
                    return False
                # Else valid.
                
            elif c1 == c3:
                # c1=c3.
                # If c2==c, set {c1, c2}. Size 2. Invalid.
                if c2 == c:
                    return False
                # If c2 != c, set {c1, c2, c}.
                # If c == c1, set {c1, c2}. Size 2. Invalid.
                if c == c1:
                    return False
                # Else valid.
                
            else:
                # c1, c2, c3 all distinct.
                # Set size is at least 3 regardless of c.
                # Even if c matches one of them, size is 3.
                # If c is new, size is 4.
                # Both valid.
                pass
                
        return True

    def greedy_solve(K):
        colors = [None] * (n + 1)
        colors[1] = 0
        for i in range(2, n + 1):
            found = False
            for c in range(K):
                if is_valid(i, c, colors):
                    colors[i] = c
                    found = True
                    break
            if not found:
                return None
        return colors[1:n+1]

    def backtrack_solve(K, time_limit):
        colors = [None] * (n + 1)
        colors[1] = 0
        
        def backtrack(i):
            if time.time() > time_limit:
                return False
            if i > n:
                return True
            
            for c in range(K):
                if is_valid(i, c, colors):
                    colors[i] = c
                    if backtrack(i + 1):
                        return True
                    colors[i] = None
            return False
            
        if backtrack(2):
            return colors[1:n+1]
        return None

    # Try Greedy for K=3
    res = greedy_solve(3)
    if res:
        return res

    # Try Greedy for K=4
    res = greedy_solve(4)
    if res:
        return res

    # Try Backtrack for K=4 with 10s limit
    if time.time() < deadline - 10:
        res = backtrack_solve(4, deadline - 10)
        if res:
            return res

    # Fallback: Valid coloring with 8 colors? 
    # c(i) = i % 8 is not guaranteed valid for 4-APs.
    # c(i) = i is always valid.
    # To get a better score, let's try Greedy for K=5
    res = greedy_solve(5)
    if res:
        return res
        
    res = greedy_solve(6)
    if res:
        return res

    # Absolute fallback: Distinct colors
    return [i for i in range(1, n+1)]
