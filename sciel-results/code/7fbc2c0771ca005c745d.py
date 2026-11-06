import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin
    
    best_set = []
    best_len = 0
    
    # Precompute "pair conflict" count for ordering
    # x is dangerous if many a,b exist with a+b=2x
    conflict_counts = [0] * 513
    for x in range(1, 513):
        lo = max(1, 2*x - 512)
        hi = x - 1
        if lo <= hi:
            conflict_counts[x] = hi - lo + 1
    
    candidates_base = list(range(1, 513))
    candidates_sorted = sorted(candidates_base, key=lambda x: conflict_counts[x])
    
    # Helper to check if cand can be added to current_set given current dp
    def _check_and_add(cand, current_set, current_set_set, dp):
        n = len(current_set)
        if n >= 48:
            return False
        if cand in current_set_set:
            return False
        
        # Check 1: Is cand the mean of any subset of current_set?
        # Need k >= 2 such that sum = k * cand is in dp[k]
        valid = True
        for k in range(2, n + 1):
            target = k * cand
            if (dp[k] >> target) & 1:
                valid = False
                break
        
        if valid:
            # Check 2: Does adding cand make any a in current_set the mean?
            # Subset T of current_set (size k>=1) + {cand} has mean a.
            # Sum(T) + cand = a * (k + 1)
            # Sum(T) = a * (k + 1) - cand
            for a in current_set:
                for k in range(1, n + 1):
                    target = a * (k + 1) - cand
                    if target < 0:
                        continue
                    if (dp[k] >> target) & 1:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            # Add cand to set and update dp
            current_set.append(cand)
            current_set_set.add(cand)
            for k in range(n + 1, 0, -1):
                dp[k] |= (dp[k - 1] << cand)
            return True
        
        return False

    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Randomize the order slightly or use different strategies
        if restart == 0:
            candidates = candidates_sorted
        elif restart % 2 == 0:
            candidates = candidates_sorted[:]
            # Shuffle the first 50% to introduce variation
            half = len(candidates) // 2
            random.shuffle(candidates[:half])
        else:
            candidates = list(candidates_base)
            random.shuffle(candidates)
            
        current_set = []
        current_set_set = set()
        
        # Incremental DP
        dp = [0] * 49
        dp[0] = 1
        
        for cand in candidates:
            if time.time() > deadline - 2.0:
                break
            if len(current_set) >= 48:
                break
                
            if _check_and_add(cand, current_set, current_set_set, dp):
                if len(current_set) > best_len:
                    best_set = list(current_set)
                    best_len = len(current_set)
                    
                    if best_len == 48:
                        return best_set
                        
    # Polish the best set found by trying to add any remaining numbers
    if best_len < 48 and best_set:
        best_set_set = set(best_set)
        # Rebuild dp for best_set
        dp = [0] * 49
        dp[0] = 1
        n = 0
        for x in best_set:
            for k in range(n + 1, 0, -1):
                dp[k] |= (dp[k - 1] << x)
            n += 1
        
        # Try to add remaining numbers
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        remaining.sort(key=lambda x: conflict_counts[x])
        
        for cand in remaining:
            if time.time() > deadline - 1.0:
                break
            if len(best_set) >= 48:
                break
            if _check_and_add(cand, best_set, best_set_set, dp):
                best_len = len(best_set)
                
    return best_set
