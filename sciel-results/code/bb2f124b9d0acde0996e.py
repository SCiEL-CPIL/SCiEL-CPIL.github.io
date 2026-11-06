import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    n = 512
    target_size = 48
    
    # We will try to build the set greedily.
    # To improve success rate, we can try multiple random orders or specific heuristics.
    # However, a deterministic increasing order is a strong baseline.
    # Let's combine: try increasing, then if fails, try random permutations.
    
    def build_dp(arr):
        """
        Build DP table for subset sums by size.
        arr: list of integers
        returns: list of bitsets, dp[k] is bitset of sums of subsets of size k
        """
        m = len(arr)
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << val)
        return dp

    def is_valid_addition(S, dp, dp_without, x):
        """
        Check if x can be added to S.
        S: current set (list)
        dp: DP table for S
        dp_without: dict {y: dp_table_for_S_minus_y}
        x: candidate element
        """
        m = len(S)
        
        # Check 1: Is x the mean of any subset of S of size k >= 2?
        # x = (sum of subset) / k  => sum = k * x
        for k in range(2, m + 1):
            target_sum = k * x
            # Max possible sum of k elements from S is k * max(S), but we use bitset bounds
            # Bitset length is fixed by max possible sum of S
            # If target_sum exceeds the highest bit in dp[k], it's 0
            if target_sum < 0:
                continue
            # We need to ensure target_sum is within the bitset range
            # The bitset dp[k] has bits up to sum(S). 
            # If target_sum > sum(S), the bit is 0.
            if (dp[k] >> target_sum) & 1:
                return False

        # Check 2: Does adding x cause any existing y in S to become the mean of a subset containing x?
        # Subset T contains x. Let U = T \ {x}. U is subset of S \ {y}.
        # y = (sum(U) + x) / (|U| + 1)
        # Let j = |U|. j >= 1 (since subset size |T| >= 2 and x is one, so U has at least 1? 
        # Wait. If |T|=2, U has size 1. If |T|=1, U has size 0, but |T| must be >=2.
        # So j ranges from 1 to m-1 (since y is not in U, U is subset of S\{y} which has size m-1).
        for y in S:
            dpy = dp_without[y]
            # dpy has size m (indices 0..m-1) corresponding to subsets of S\{y}
            for j in range(1, m):
                # Target sum for U: sum(U) = (j + 1) * y - x
                target_sum = (j + 1) * y - x
                if target_sum < 0:
                    continue
                # Check if dpy[j] has bit target_sum set
                if (dpy[j] >> target_sum) & 1:
                    return False
        
        return True

    def add_element(S, dp, dp_without, x):
        """
        Add x to S and update DP structures.
        """
        m = len(S)
        
        # Store dp for S\{x} which is just the current dp (before adding x)
        # But we need to store it for x as well? 
        # dp_without[x] will be the dp of S (old).
        dp_without[x] = list(dp)
        
        # Update dp_without for existing elements y
        # New dp for S \ {y} includes x.
        # Old dp for S \ {y} is dpy (size m, indices 0..m-1).
        # New S \ {y} has size m. 
        # We need to add x to the subsets of S \ {y}.
        for y in S:
            dpy = dp_without[y]
            # Extend dpy to size m+1? No, S\{y} now has m elements.
            # Old S\{y} had m-1 elements.
            # dpy currently has indices 0..m-1.
            # We need to create new_dpy of size m+1 (indices 0..m).
            new_dpy = dpy + [0]
            # Add x to subsets
            # Iterate k from m down to 1
            # new_dpy[k] |= (new_dpy[k-1] << x)
            for k in range(m, 0, -1):
                new_dpy[k] |= (new_dpy[k - 1] << x)
            dp_without[y] = new_dpy
            
        # Update main dp for S
        # Old dp has size m+1 (indices 0..m).
        # New dp has size m+2 (indices 0..m+1).
        new_dp = dp + [0]
        for k in range(m + 1, 0, -1):
            new_dp[k] |= (new_dp[k - 1] << x)
            
        S.append(x)
        return S, new_dp, dp_without

    best_set = []
    
    # Try a few strategies
    # Strategy 1: Increasing order
    candidates = list(range(1, n + 1))
    
    # Strategy 2: Randomized restarts if needed
    # But first, try increasing order. It's fast and often works well for this type of constraint.
    
    S = []
    dp = [1]
    dp_without = {}
    
    for x in candidates:
        if len(S) >= target_size:
            break
        if time.time() > deadline:
            break
            
        if is_valid_addition(S, dp, dp_without, x):
            S, dp, dp_without = add_element(S, dp, dp_without, x)
            
    if len(S) > len(best_set):
        best_set = list(S)
        
    # If we didn't reach target, try random orders
    if len(best_set) < target_size:
        rng = random.Random(42)
        max_restarts = 50
        
        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            order = list(range(1, n + 1))
            rng.shuffle(order)
            
            S = []
            dp = [1]
            dp_without = {}
            
            for x in order:
                if len(S) >= target_size:
                    break
                if time.time() > deadline - 2.0:
                    break
                    
                if is_valid_addition(S, dp, dp_without, x):
                    S, dp, dp_without = add_element(S, dp, dp_without, x)
                    
            if len(S) > len(best_set):
                best_set = list(S)
                if len(best_set) >= target_size:
                    break

    return best_set
