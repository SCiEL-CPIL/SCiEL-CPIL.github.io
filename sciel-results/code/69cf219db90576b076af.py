import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    best_len = 0
    
    # Precompute "pair conflict" count for ordering
    # x is dangerous if many a,b exist with a+b=2x
    # This is a static property of x in [1, 512]
    conflict_counts = [0] * 513
    for x in range(1, 513):
        # a + b = 2x, a < b, a,b in [1, 512]
        # a can range from max(1, 2x-512) to min(x-1, 512-x) ... wait
        # a < 2x - a => 2a < 2x => a < x
        # b = 2x - a <= 512 => a >= 2x - 512
        # a >= 1
        lo = max(1, 2*x - 512)
        hi = x - 1
        if lo <= hi:
            conflict_counts[x] = hi - lo + 1
            
    # Sort candidates by conflict count (ascending)
    # Numbers with fewer 2-sum partners are "safer" to include early?
    # Actually, if we include safe numbers, we might block dangerous ones?
    # Or if we include dangerous ones, we block many others?
    # Usually, greedy picks the "most constrained" or "least constrained".
    # Let's try least constrained first (lowest conflict count).
    
    candidates_base = list(range(1, 513))
    candidates_sorted = sorted(candidates_base, key=lambda x: conflict_counts[x])
    
    max_restarts = 200 # We can do more restarts now due to speedup
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Randomize the order slightly or use different strategies
        if restart == 0:
            candidates = candidates_sorted
        elif restart % 2 == 0:
            candidates = candidates_sorted
            # Shuffle the first 50% to introduce variation
            half = len(candidates) // 2
            random.shuffle(candidates[:half])
        else:
            candidates = list(candidates_base)
            random.shuffle(candidates)
            
        current_set = []
        current_set_set = set()
        
        # Incremental DP
        # dp[k] is a bitset of sums achievable by subsets of size k
        # Max size 48. Max sum 48*512 = 24576.
        # We initialize dp as we add elements.
        # Since we are adding one by one, we can maintain dp.
        
        # However, the check for "does adding cand break existing a?" requires dp of S.
        # The check for "is cand mean of S?" requires dp of S.
        # So we use the current dp of S.
        
        dp = [0] * 49  # Index 0 to 48
        dp[0] = 1  # Sum 0 with 0 elements
        
        for cand in candidates:
            if time.time() > deadline - 2.0:
                break
                
            if cand in current_set_set:
                continue
                
            n = len(current_set)
            if n >= 48:
                break
                
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
                # Check if Sum(T) is in dp[k]
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
                
                # Update dp: for k from n+1 down to 1
                # dp[k] |= dp[k-1] << cand
                for k in range(n + 1, 0, -1):
                    dp[k] |= (dp[k - 1] << cand)
                
                if len(current_set) > best_len:
                    best_set = list(current_set)
                    best_len = len(current_set)
                    
                    if best_len == 48:
                        return best_set
                        
    # If we didn't reach 48, try to polish the best set found
    # by trying to add any remaining numbers
    if best_len < 48:
        best_set_set = set(best_set)
        # Rebuild dp for best_set
        dp = [0] * 49
        dp[0] = 1
        for x in best_set:
            n = len(best_set) # This is wrong, n changes
            # We need to rebuild properly
            pass
        
        # Simpler: Just try to add remaining numbers to best_set using the same check logic
        # But we need the DP for best_set.
        # Let's just do a few more random restarts if time permits.
        
    return best_set
