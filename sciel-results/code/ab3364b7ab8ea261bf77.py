import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Target differences: 1..30
    # Bitmask for target differences: bits 1 to 30 set
    TARGET_MASK = 0
    for d in range(1, 31):
        TARGET_MASK |= (1 << d)
    
    # We need to find a Sidon set S containing 0, with max(S) minimized,
    # such that all differences 1..30 are present in S.
    
    # Bitmask backtracking
    # State: current set (list), used_diffs_mask (int), last_elem (int)
    # We want to add elements x > last_elem such that:
    # 1. x - s is not in used_diffs_mask for any s in S
    # 2. x - s1 != x - s2 for any s1 != s2 in S (distinct differences from x)
    # 3. We eventually cover TARGET_MASK
    
    def search(D):
        """
        Search for a valid Sidon set within [0, D].
        Returns the set if found, else None.
        """
        # We start with {0}
        current_set = [0]
        used_mask = 0
        
        def dfs(last_elem):
            nonlocal current_set, used_mask
            
            # Check time
            if time.time() > deadline:
                return None
            
            # Check if we have covered all target differences
            if (used_mask & TARGET_MASK) == TARGET_MASK:
                return current_set[:]
            
            # Pruning: 
            # If we have too many elements, stop? Max 32.
            if len(current_set) >= 32:
                return None
            
            # Try candidates x from last_elem + 1 to D
            # Optimization: We can try candidates in an order that likely covers missing diffs.
            # However, for minimal D, we should probably just try all valid x.
            # To speed up, we can precompute which diffs are missing.
            missing_mask = TARGET_MASK & ~used_mask
            if missing_mask == 0:
                return current_set[:]
            
            best_result = None
            
            # Generate candidates
            # To ensure we find the smallest max element, we iterate D upwards.
            # Inside a fixed D, the order of x doesn't affect the max element (which is D or less),
            # but we want to find *a* solution.
            
            for x in range(last_elem + 1, D + 1):
                # Check conflicts
                new_diffs_mask = 0
                valid = True
                
                for s in current_set:
                    d = x - s
                    # d must be positive (guaranteed since x > last_elem >= s)
                    # Check if d is already used
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    # Check if d conflicts with other new diffs
                    if (new_diffs_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                
                if not valid:
                    continue
                
                # Heuristic: Prioritize x that covers more missing diffs?
                # For now, just recurse. The search space is small for small D.
                
                current_set.append(x)
                old_mask = used_mask
                used_mask = old_mask | new_diffs_mask
                
                res = dfs(x)
                if res is not None:
                    return res
                
                # Backtrack
                current_set.pop()
                used_mask = old_mask
            
            return None
        
        return dfs(0)

    # Iterate D from lower bound to upper bound
    # Lower bound: 30 is a difference, so max element >= 30.
    # Also, we need enough elements.
    # Let's start from 30.
    for D in range(30, 100):
        if time.time() > deadline - 10:
            break
            
        res = search(D)
        if res is not None:
            # Verify solution
            S = res
            if len(S) > 32: continue
            if 0 not in S: continue
            if max(S) > 10**6: continue
            
            # Check Sidon and Coverage
            diffs = set()
            valid = True
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    d = abs(S[i] - S[j])
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                continue
                
            if all(d in diffs for d in range(1, 31)):
                return S
                
    # Fallback: Greedy construction if search fails (should not happen for D < 100)
    S = [0]
    used_mask = 0
    for _ in range(32):
        if time.time() > deadline - 5:
            break
        missing = [d for d in range(1, 31) if not (used_mask >> d) & 1]
        if not missing:
            break
            
        best_x = None
        best_score = -1
        
        # Try x up to 1000
        for x in range(1, 1000):
            if x in S: continue
            new_diffs_mask = 0
            valid = True
            for s in S:
                d = abs(x - s)
                if (used_mask >> d) & 1:
                    valid = False
                    break
                if (new_diffs_mask >> d) & 1:
                    valid = False
                    break
                new_diffs_mask |= (1 << d)
            
            if not valid:
                continue
            
            # Score: number of missing diffs covered
            score = 0
            for d in missing:
                if (new_diffs_mask >> d) & 1:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
                
        if best_x is None:
            # Pick smallest valid x
            for x in range(1, 1000):
                if x in S: continue
                new_diffs_mask = 0
                valid = True
                for s in S:
                    d = abs(x - s)
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    if (new_diffs_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                if valid:
                    best_x = x
                    break
        
        if best_x is None:
            break
            
        S.append(best_x)
        S.sort()
        
        # Update used_mask
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = S[j] - S[i]
                used_mask |= (1 << d)
                
    return S
