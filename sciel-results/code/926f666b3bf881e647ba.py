import sys
import time

def construct():
    """
    Construct a Sidon set containing 0 with all differences 1..30, minimizing max element.
    Uses optimized Backtracking with Bitmasks and Strong Pruning.
    """
    TARGET_MASK = (1 << 30) - 1  # Bits 0..29 represent diffs 1..30
    
    def solve(M):
        # S: list of elements
        # D_set: set of all differences
        # D_mask: bitmask of differences in 1..30
        # last: last element added
        
        def backtrack(S, D_set, D_mask, last):
            # Check success
            if (D_mask & TARGET_MASK) == TARGET_MASK:
                return S[:]
            
            # Pruning: Not enough capacity
            if len(S) >= 32:
                return None
            
            missing_mask = TARGET_MASK & ~D_mask
            num_missing = bin(missing_mask).count('1')
            
            if num_missing == 0:
                return S[:]
            
            k_max = 32 - len(S)
            # Max new diffs from k_max new elements
            potential_new = k_max * len(S) + k_max * (k_max - 1) // 2
            if num_missing > potential_new:
                return None
            
            # Try adding x > last
            for x in range(last + 1, M + 1):
                # Check Sidon property
                valid = True
                new_diffs = []
                for s in S:
                    d = x - s
                    if d in D_set:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                new_D_set = D_set.union(new_diffs)
                
                new_D_mask = D_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_D_mask |= (1 << (d - 1))
                
                result = backtrack(S + [x], new_D_set, new_D_mask, x)
                if result is not None:
                    return result
            
            return None
        
        return backtrack([0], set(), 0, 0)

    start_time = time.time()
    max_time = 110
    
    # Lower bound for M
    # N=9, diffs=36. M>=36.
    for M in range(36, 100):
        if time.time() - start_time > max_time:
            break
        
        sol = solve(M)
        if sol is not None:
            # Verify
            if 0 in sol and len(set(sol)) == len(sol):
                diffs = set()
                n = len(sol)
                sol = sorted(sol)
                for i in range(n):
                    for j in range(i + 1, n):
                        diffs.add(sol[j] - sol[i])
                if all(d in diffs for d in range(1, 31)) and len(diffs) == n*(n-1)//2:
                    return sol

    # Fallback
    a = [0]
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for d in range(1, 31):
        if d in set(get_diffs(sorted(a))):
            continue
        found = False
        candidates = set()
        for s in a:
            candidates.add(s + d)
        
        for x in sorted(candidates):
            if x < 0 or x in a:
                continue
            trial = sorted(a + [x])
            d_list = get_diffs(trial)
            if len(set(d_list)) == len(d_list):
                a = trial
                found = True
                break
        if not found:
            y = max(a) + 100
            while True:
                trial = sorted(a + [y])
                d_list = get_diffs(trial)
                if len(set(d_list)) == len(d_list):
                    a = trial
                    break
                y += 1
    return a
