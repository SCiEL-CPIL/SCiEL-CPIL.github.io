import sys
import time

def construct():
    max_val = 512
    max_count = 48
    
    # Bitset DP
    selected = []
    dp = [0] * (max_count + 1)
    dp[0] = 1
    
    dp_without = {}
    
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    for x in range(1, max_val + 1):
        if time.time() > deadline:
            break
            
        valid = True
        
        # Check 1: x is mean of subset of selected
        for k in range(2, len(selected) + 1):
            target_sum = k * x
            if target_sum < 0:
                continue
            if (dp[k] >> target_sum) & 1:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check 2: any a in selected is mean of subset containing x
        for a in selected:
            dp_a = dp_without[a]
            for j in range(0, len(selected)):
                target_sum = (j + 1) * a - x
                if target_sum < 0:
                    continue
                if (dp_a[j] >> target_sum) & 1:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Save old dp for dp_without[x]
        old_dp = dp[:]
        
        # Update dp with x
        for k in range(len(selected), -1, -1):
            dp[k + 1] |= dp[k] << x
            
        # Update dp_without for existing elements
        for a in selected:
            dp_a = dp_without[a]
            for k in range(len(selected), -1, -1):
                dp_a[k + 1] |= dp_a[k] << x
        
        # Set dp_without[x] to old_dp
        dp_without[x] = old_dp
        
        selected.append(x)
        
    return selected
