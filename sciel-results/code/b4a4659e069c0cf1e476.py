import math
from functools import lru_cache

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    n = P['n']
    # LCM of 1..12 is 27720
    # We only need to check m in [0, 27719]
    # Because the pattern of multiples repeats every 27720
    # f(n, m) depends on m mod LCM(1..n)
    
    # Calculate LCM of 1..12
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    LCM = 1
    for i in range(1, n + 1):
        LCM = lcm(LCM, i)
        
    best_m = 0
    best_L = -1
    
    # Precompute indices
    indices = list(range(1, n + 1))
    
    for m in range(LCM):
        # Find the first multiple of i > m for each i
        # x_i is the smallest multiple of i such that x_i > m
        # x_i = ((m // i) + 1) * i
        first_multiples = [((m // i) + 1) * i for i in indices]
        
        # The minimum L must be at least max(x_i - m)
        min_L = max(x - m for x in first_multiples)
        
        L = min_L
        # We need to find the smallest L >= min_L such that a matching exists
        # We can iterate L upwards
        # Upper bound for L? The prompt says max 144, but let's be safe and go up to, say, 200
        # If no match is found, something is wrong, but theoretically it will always be found eventually
        
        while True:
            # Check if matching exists for current L
            # Available candidates for i: {k in (m, m+L] | i divides k}
            # This is {x_i, x_i + i, ...} <= m+L
            
            # Build list of candidates for each i
            candidates = []
            for i in indices:
                start_val = first_multiples[i-1]
                cand_list = []
                val = start_val
                while val <= m + L:
                    cand_list.append(val)
                    val += i
                candidates.append(cand_list)
            
            # Check for perfect matching using backtracking
            # To optimize, sort indices by number of candidates (ascending)
            # But we need to map back to original indices or just use the list of lists
            # Let's create a list of (index, candidates) and sort by len(candidates)
            sorted_cands = sorted([(i, c) for i, c in enumerate(candidates)], key=lambda x: len(x[1]))
            
            # If any list is empty, no match
            if any(len(c) == 0 for _, c in sorted_cands):
                L += 1
                continue
                
            # Backtracking
            used = set()
            
            def backtrack(k):
                if k == n:
                    return True
                i, cands = sorted_cands[k]
                for c in cands:
                    if c not in used:
                        used.add(c)
                        if backtrack(k + 1):
                            return True
                        used.remove(c)
                return False
            
            if backtrack(0):
                # Found a valid L for this m
                if L > best_L:
                    best_L = L
                    best_m = m
                break
            else:
                L += 1
                # Safety break if L gets too large
                if L > 200:
                    break
                    
    return int(best_m)
