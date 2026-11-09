import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial valid tail
    S_tail = set(range(171, 257))
    
    # Precompute conflicts with tail for x in 1..170
    # conflict_tail[x] is True if there exist b, c in S_tail such that x < b <= c and x | (b+c)
    conflict_tail = [False] * 171
    # Optimize: b+c ranges from 172+172=344 to 256+255=511
    # For a given x, we check multiples k*x in [344, 511]
    # and see if k*x can be written as b+c with b,c in S_tail, b>c? No b<=c.
    # b = k*x - c.
    # Conditions:
    # 1. c in S_tail (171..256)
    # 2. b in S_tail (171..256)
    # 3. b <= c
    # 4. x < b (automatically true if b>=171 and x<=170)
    
    for x in range(1, 171):
        max_k = 511 // x
        for k in range(2, max_k + 1):
            s = k * x
            if s < 344:
                continue
            if s > 511:
                break
            # Check if s = b + c with b, c in [171, 256] and b <= c
            # c >= s/2
            # c <= 256
            # b = s - c >= 171 => c <= s - 171
            c_min = max(171, (s + 1) // 2)
            c_max = min(256, s - 171)
            if c_min <= c_max:
                # There exists a valid pair
                conflict_tail[x] = True
                break
                
    S_small = set()
    
    # Greedy addition
    for x in range(1, 171):
        if time.time() > deadline - 5:
            break
            
        if conflict_tail[x]:
            continue
            
        # Check Case 2: a in S_small, c in S_tail, a < x <= c
        # a | (x + c) ?
        valid = True
        for a in S_small:
            # a < x is guaranteed
            # c in S_tail (171..256)
            # x + c range: 1+171=172 to 170+256=426
            # Check if x+c is multiple of a
            # (x+c) % a == 0
            # c = k*a - x
            # 171 <= k*a - x <= 256
            # 171 + x <= k*a <= 256 + x
            k_min = (171 + x + a - 1) // a
            k_max = (256 + x) // a
            for k in range(k_min, k_max + 1):
                c = k * a - x
                if 171 <= c <= 256:
                    # c is in S_tail
                    # Check b <= c? Here x is b, c is c.
                    # Condition for Case 2: a < x <= c.
                    # We have a < x.
                    # Need x <= c.
                    if x <= c:
                        valid = False
                        break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Check Case 3: a, b in S_small, a < b <= x
        # a | (b + x) ?
        for a in S_small:
            for b in S_small:
                if a < b < x:
                    if (b + x) % a == 0:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            S_small.add(x)
            
    result = list(S_tail | S_small)
    return result
