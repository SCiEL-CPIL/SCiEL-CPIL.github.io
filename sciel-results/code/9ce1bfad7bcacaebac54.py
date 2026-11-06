import numpy as np
import random
import time

def construct():
    start_time = time.time()
    max_time = 85.0
    n = 512
    max_size = 48
    # Max sum is 48 * 512 = 24576
    max_sum = max_size * n + 100 
    
    best_set = []
    best_size = 0
    
    # Pre-allocate DP arrays
    # dp[k] is a bool array of size max_sum + 1
    # We need to reset them for each restart
    
    # To speed up restarts, we can just re-initialize the arrays
    # But creating 49 arrays of size 24676 is not too bad.
    
    for _ in range(500): # Number of restarts
        if time.time() - start_time > max_time:
            break
            
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = []
        # Initialize DP
        # dp[k][s]
        dp = [np.zeros(max_sum + 1, dtype=bool) for _ in range(max_size + 1)]
        dp[0][0] = True
        
        current_max_k = 0
        
        for x in candidates:
            if time.time() - start_time > max_time:
                break
                
            # Check validity
            # 1. x is not the mean of any subset of S (size >= 2)
            valid = True
            for k in range(2, current_max_k + 1):
                target = k * x
                if target > max_sum:
                    break
                if dp[k][target]:
                    valid = False
                    break
            
            # 2. No existing element y in S is the mean of any subset of S U {x}
            #    Subset must include x. Let T be subset of S (size k >= 1) such that
            #    mean(T U {x}) = y
            #    sum(T) + x = (k+1)*y  => sum(T) = (k+1)*y - x
            if valid:
                for y in S:
                    for k in range(1, current_max_k + 1):
                        target = (k + 1) * y - x
                        if target < 0 or target > max_sum:
                            continue
                        if dp[k][target]:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Update DP in place
                # Iterate k from current_max_k down to 0
                # dp[k+1] |= shift(dp[k], x)
                for k in range(current_max_k, -1, -1):
                    if x <= max_sum:
                        # Shift dp[k] left by x
                        # dp[k+1] |= dp[k] shifted
                        # We need to be careful with bounds
                        # shifted = np.zeros_like(dp[k])
                        # shifted[x:] = dp[k][:-x]
                        # dp[k+1] |= shifted
                        
                        # In-place shift logic:
                        # We cannot simply do dp[k+1][x:] |= dp[k][:-x] because
                        # dp[k+1] is being updated, but dp[k] is read-only in this step.
                        # This is safe.
                        dp[k+1][x:] |= dp[k][:-x]
                
                S.append(x)
                current_max_k += 1
                
                if current_max_k == max_size:
                    break # Reached max size
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
