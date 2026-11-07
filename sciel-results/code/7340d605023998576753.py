import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    max_val = 256
    max_size = 60
    
    def is_valid_addition(dp, x, n_current):
        n_new = n_current + 1
        new_sums = [0] * (n_new + 1)
        
        # Calculate new sums
        for k in range(1, n_new + 1):
            new_sums[k] = (dp[k-1] << x)
            
        # Check new-new collisions
        for k in range(1, n_new + 1):
            nk = new_sums[k]
            if nk == 0: continue
            for m in range(k+1, n_new + 1):
                nm = new_sums[m]
                if nm == 0: continue
                if nk & nm:
                    return False
                    
        # Check new-old collisions
        # Precompute old_union
        old_union = 0
        for m in range(n_current + 1):
            old_union |= dp[m]
            
        for k in range(1, n_new + 1):
            nk = new_sums[k]
            if nk == 0: continue
            
            # We need nk to be disjoint from dp[m] for all m != k
            # This is equivalent to nk being disjoint from (old_union \ dp[k])
            # Note: if k > n_current, dp[k] is 0, so old_union \ dp[k] is just old_union.
            
            # old_union ^ (old_union & dp[k]) is old_union without bits in dp[k]
            forbidden = old_union & ~dp[k]
            if nk & forbidden:
                return False
                
        return True

    best_set = []
    
    # Strategy 1: Greedy Smallest
    # This is often the best for this problem type.
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        dp = [0] * (max_size + 1)
        dp[0] = 1
        current_set = []
        
        if attempt == 0:
            candidates = list(range(1, max_val + 1))
        elif attempt % 2 == 1:
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
        else:
            # Try skipping small numbers?
            candidates = list(range(1, max_val + 1))
            # Swap first few
            for i in range(min(5, len(candidates)-1)):
                candidates[i], candidates[i+1] = candidates[i+1], candidates[i]
                
        for x in candidates:
            if len(current_set) >= max_size:
                break
            
            if is_valid_addition(dp, x, len(current_set)):
                current_set.append(x)
                # Update dp
                for k in range(len(current_set), 0, -1):
                    dp[k] |= (dp[k-1] << x)
                    
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            if len(best_set) == max_size:
                return best_set
                
    return best_set
