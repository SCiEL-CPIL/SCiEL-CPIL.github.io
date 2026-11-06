import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 512
    
    def is_valid_fast(S):
        S_list = list(S)
        n = len(S_list)
        if n < 3:
            return True
        
        max_sum = sum(S_list)
        
        for i, x in enumerate(S_list):
            others = S_list[:i] + S_list[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            # dp[k] is a bitmask where bit j is 1 if sum j is achievable with k elements
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for val in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << val)
            
            for k in range(2, m + 1):
                target = k * x
                if (dp[k] >> target) & 1:
                    return False
        return True

    def has_3term_ap(S, x):
        for y in S:
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if z in S and z != x and z != y:
                    return True
            z = 2 * x - y
            if z in S and z != x and z != y:
                return True
            z = 2 * y - x
            if z in S and z != x and z != y:
                return True
        return False

    # 1. Generate Ternary Set
    ternary_set = set()
    for i in range(1, N + 1):
        t = i
        ok = True
        while t > 0:
            if t % 3 == 1:
                ok = False
                break
            t //= 3
        if ok:
            ternary_set.add(i)
            
    # Verify ternary set
    if is_valid_fast(ternary_set):
        best_set = ternary_set
    else:
        # If ternary set is invalid (unlikely for AP-free), build greedily
        best_set = set()
        
    # 2. Try to augment best_set
    # Sort candidates by "safety" (fewest conflicts)
    candidates = [x for x in range(1, N + 1) if x not in best_set]
    
    # Calculate conflict degree for sorting
    def get_degree(x, S):
        count = 0
        for y in S:
            if has_3term_ap({x, y}, x): # Just check if x conflicts with y in S?
                 pass
        # Simpler: just count how many z in S make x,y,z an AP
        # We can just use the order from before or random
        return 0
        
    # Let's just try adding in increasing order first, then random
    for x in sorted(candidates):
        if time.time() > deadline:
            break
        if has_3term_ap(best_set, x):
            continue
        best_set.add(x)
        if is_valid_fast(best_set):
            pass
        else:
            best_set.remove(x)
            
    # Try random additions
    random.shuffle(candidates)
    for x in candidates:
        if time.time() > deadline:
            break
        if x in best_set:
            continue
        if has_3term_ap(best_set, x):
            continue
        best_set.add(x)
        if is_valid_fast(best_set):
            pass
        else:
            best_set.remove(x)

    return list(best_set)
