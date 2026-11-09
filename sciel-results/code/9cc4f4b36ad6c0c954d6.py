import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Generate Cantor-like set (ternary digits 0,1) <= 512
    candidates = []
    for i in range(513):
        s = ''
        n = i
        while n > 0:
            s = str(n % 3) + s
            n //= 3
        if s == '':
            s = '0'
        if all(c in '01' for c in s):
            candidates.append(i)
            
    # Ensure 0 is not included (positive integers)
    candidates = [c for c in candidates if c > 0]
    
    # Sort candidates
    candidates.sort()
    
    # If we need more, add others? 
    # For now, just use candidates. If < 48, we might need to expand.
    # But 64 candidates should be enough if we remove few.
    
    selected = []
    
    # Helper to check if x is mean of subset of S (S does not contain x)
    # Using DP
    def is_mean_of_subset(x, S):
        # Check if k*x is sum of k elements from S for k >= 2
        # DP[k] is bitset of sums using k elements
        max_sum = sum(S)
        # Bitset DP
        # dp[k] has bit s set if s is sum of k elements
        dp = [0] * (len(S) + 1)
        dp[0] = 1 # bit 0 set
        for val in S:
            # Update in reverse to avoid using same element twice
            for k in range(len(S), 0, -1):
                dp[k] |= (dp[k-1] << val)
        
        for k in range(2, len(S) + 1):
            if dp[k] >> (k * x) & 1:
                return True
        return False

    def check_validity_with_c(c, S):
        # Check if adding c to S keeps it valid.
        # 1. c is not mean of subset of S
        if is_mean_of_subset(c, S):
            return False
            
        # 2. No x in S becomes mean of subset of S U {c} involving c
        # For each x in S, check if k*x is sum of k elements from S U {c} \ {x}
        # Such a sum must include c.
        # So k*x - c must be sum of k-1 elements from S \ {x}
        
        # We can compute dp for S once?
        # But we need dp for S \ {x}.
        
        # Optimization:
        # If S is small, we can just check.
        # Let's compute dp_S for S.
        # Then for each x, we need to know if k*x - c is in dp_{S\{x}}[k-1].
        
        # To avoid recomputing DP for each x, we can note that:
        # dp_{S\{x}} is derived from dp_S by removing x.
        # This is hard with bitsets.
        
        # Alternative: Just compute DP for S \ {x} for each x.
        # |S| <= 48.
        # Cost: |S| * DP_Cost.
        
        for x in S:
            S_excl_x = [y for y in S if y != x]
            if not S_excl_x:
                continue
            # Check if k*x - c is sum of k-1 elements from S_excl_x
            # k ranges from 2 to |S| (since S U {c} has size |S|+1, max k is |S|)
            # Actually, subset size k from S U {c} \ {x}.
            # Max size is |S|.
            # Min size 2.
            
            # Compute DP for S_excl_x
            max_sum_excl = sum(S_excl_x)
            dp_excl = [0] * (len(S_excl_x) + 1)
            dp_excl[0] = 1
            for val in S_excl_x:
                for k in range(len(S_excl_x), 0, -1):
                    dp_excl[k] |= (dp_excl[k-1] << val)
            
            for k in range(2, len(S) + 1):
                target_sum = k * x - c
                if target_sum < 0:
                    continue
                # We need sum of k-1 elements from S_excl_x to equal target_sum
                if k-1 <= len(S_excl_x):
                    if dp_excl[k-1] >> target_sum & 1:
                        return False
                        
        return True

    for c in candidates:
        if len(selected) >= 48:
            break
        if time.time() > deadline:
            break
            
        if check_validity_with_c(c, selected):
            selected.append(c)
            
    # If we didn't reach 48, try adding other numbers
    if len(selected) < 48:
        # Generate remaining numbers
        remaining = [i for i in range(1, 513) if i not in selected]
        remaining.sort()
        for c in remaining:
            if len(selected) >= 48:
                break
            if time.time() > deadline:
                break
            if check_validity_with_c(c, selected):
                selected.append(c)
                
    return selected

# Test locally?
# print(construct())
