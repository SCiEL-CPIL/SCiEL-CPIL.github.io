# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = []
    best_len = 0
    
    # Precompute 3-term AP conflict: for each x, count pairs (a,c) with a+c=2x
    # This is the k=2 case, the most common blocker
    ap_conflicts = [0] * 513
    for x in range(1, 513):
        lo = max(1, 2*x - 512)
        hi = x - 1
        if lo <= hi:
            ap_conflicts[x] = hi - lo + 1
    
    # Sort by AP conflict count (ascending) - numbers with fewer 3-term AP partners first
    candidates_sorted = sorted(range(1, 513), key=lambda x: ap_conflicts[x])
    
    max_restarts = 300
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Choose candidate ordering strategy
        if restart == 0:
            candidates = list(candidates_sorted)
        elif restart == 1:
            candidates = list(reversed(candidates_sorted))
        elif restart == 2:
            candidates = list(range(1, 513))
            random.shuffle(candidates)
        elif restart % 4 == 0:
            # Perturbed sorted order
            candidates = list(candidates_sorted)
            for _ in range(30):
                i = random.randint(0, len(candidates) - 2)
                candidates[i], candidates[i+1] = candidates[i+1], candidates[i]
        elif restart % 4 == 1:
            # Random with bias toward low AP conflict
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            candidates.sort(key=lambda x: ap_conflicts[x] + random.random() * 30)
        elif restart % 4 == 2:
            # Start with extremes, then fill middle
            candidates = list(candidates_sorted)
            random.shuffle(candidates[:len(candidates)//2])
            random.shuffle(candidates[len(candidates)//2:])
        else:
            # Pure random
            candidates = list(range(1, 513))
            random.shuffle(candidates)
        
        current_set = []
        current_set_set = set()
        
        # Incremental DP: dp[k] is bitset of sums achievable by subsets of size k
        # Max size 48, max sum 48*512 = 24576
        dp = [0] * 49
        dp[0] = 1  # Sum 0 with 0 elements
        
        for cand in candidates:
            if time.time() > deadline - 5.0:
                break
            
            if cand in current_set_set:
                continue
            
            n = len(current_set)
            if n >= 48:
                break
            
            valid = True
            
            # Check 1: Is cand the mean of any subset of current_set with size >= 2?
            # Need: exists k >= 2 such that bit (k * cand) is set in dp[k]
            for k in range(2, n + 1):
                target = k * cand
                if target <= 24576 and (dp[k] >> target) & 1:
                    valid = False
                    break
            
            if valid:
                # Check 2: For each a in current_set, is a the mean of a subset including cand?
                # U = {cand} ∪ T where T ⊆ current_set\{a}, |T| >= 1
                # cand + sum(T) = a*(|T|+1)
                # sum(T) = a*(k+1) - cand where k = |T| >= 1
                for a in current_set:
                    for k in range(1, n + 1):
                        target = a * (k + 1) - cand
                        if 0 <= target <= 24576 and (dp[k] >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Add cand to set and update dp
                current_set.append(cand)
                current_set_set.add(cand)
                
                # Update dp: for k from n+1 down to 1
                # dp[k] |= dp[k-1] << cand
                for k in range(n + 1, 0, -1):
                    dp[k] |= (dp[k - 1] << cand)
                
                if len(current_set) > best_len:
                    best_set = list(current_set)
                    best_len = len(current_set)
                    
                    if best_len >= 48:
                        return best_set
    
    # Polish pass: try to add remaining numbers to best_set
    if best_set and best_len < 48:
        best_set_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        remaining.sort(key=lambda x: ap_conflicts[x])
        
        # Rebuild dp for best_set
        dp = [0] * 49
        dp[0] = 1
        for x in best_set:
            n = len(best_set)
            for k in range(n, 0, -1):
                dp[k] |= (dp[k - 1] << x)
        
        for cand in remaining:
            if time.time() > deadline - 2.0:
                break
            
            n = len(best_set)
            if n >= 48:
                break
            
            valid = True
            
            # Check 1
            for k in range(2, n + 1):
                target = k * cand
                if target <= 24576 and (dp[k] >> target) & 1:
                    valid = False
                    break
            
            if valid:
                # Check 2
                for a in best_set:
                    for k in range(1, n + 1):
                        target = a * (k + 1) - cand
                        if 0 <= target <= 24576 and (dp[k] >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                best_set.append(cand)
                best_set_set.add(cand)
                for k in range(n + 1, 0, -1):
                    dp[k] |= (dp[k - 1] << cand)
                best_len = len(best_set)
                
                if best_len >= 48:
                    return best_set
    
    return best_set

# EVOLVE-BLOCK-END
