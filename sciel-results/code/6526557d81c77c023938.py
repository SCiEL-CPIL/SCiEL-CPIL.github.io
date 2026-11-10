import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    def get_dp(S):
        """
        Compute DP table for subset sums by size for set S.
        dp[k] is a bitset where bit s is 1 if there is a subset of S of size k summing to s.
        """
        n = len(S)
        max_sum = sum(S)
        # dp[k] will be an integer. Bit i represents sum i.
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty subset sums to 0
        
        for x in S:
            # Update in reverse to avoid using the same element multiple times
            for k in range(n, 0, -1):
                # Shift dp[k-1] by x and OR into dp[k]
                dp[k] |= (dp[k-1] << x)
        
        return dp

    def can_add(S, x):
        """
        Check if x can be added to S without violating the condition.
        Condition: No element in S U {x} is the mean of a subset of at least 2 OTHER elements.
        """
        if not S:
            return True
        
        n = len(S)
        # We need to check two cases:
        # 1. x is the mean of a subset of S.
        # 2. Some y in S becomes the mean of a subset of S U {x} that includes x.
        
        # Compute DP for S
        # Optimization: We can cache DP for S if we are checking multiple x's against the same S?
        # No, S changes as we add elements. But we check many x's against the SAME S in a greedy step.
        # So we should compute DP for S once per greedy step iteration, not per candidate.
        
        pass # Placeholder, will restructure

    # Restructured approach:
    # In each greedy step, we have a current set S.
    # We compute DP for S once.
    # Then we iterate through candidates x.
    # For each x, we check validity using the precomputed DP.
    
    def check_validity_with_dp(dp, S, x):
        """
        dp: DP table for S.
        S: Current set.
        x: Candidate to add.
        Returns True if valid.
        """
        n = len(S)
        
        # Case 1: x is the mean of a subset of S.
        # sum(subset) = k * x, k >= 2.
        for k in range(2, n + 1):
            target_sum = k * x
            # Check if bit target_sum is set in dp[k]
            if (dp[k] >> target_sum) & 1:
                return False
        
        # Case 2: Some y in S is the mean of a subset of S U {x} including x.
        # Let the subset be T U {x}, where T is a subset of S.
        # |T U {x}| = |T| + 1 = k+1 (let |T|=k).
        # Sum(T U {x}) = Sum(T) + x.
        # Mean condition: Sum(T) + x = (k+1) * y.
        # Sum(T) = (k+1) * y - x.
        # We need to check if there exists a subset T of S of size k summing to (k+1)y - x.
        # k can range from 0 to n.
        # If k=0, T is empty, Sum(T)=0. Then 0 = y - x => x=y, impossible.
        # So k >= 1.
        
        for y in S:
            for k in range(1, n + 1):
                target_sum = (k + 1) * y - x
                if target_sum < 0:
                    continue
                # Max possible sum for k elements from S is k * max(S)
                # We can just check the bit, if target_sum is out of range of the bitset, 
                # the shift will be huge or the bit will be 0.
                # Python ints are arbitrary precision, so (dp[k] >> target_sum) & 1 is safe.
                if (dp[k] >> target_sum) & 1:
                    return False
        
        return True

    best_set = []
    max_restarts = 1000
    
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
            
        S = []
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        # We try to add elements one by one
        # To make it faster, we can filter candidates that are "likely" to work?
        # Or just iterate.
        
        # We need to recompute DP every time S changes.
        # But S changes only when we successfully add an element.
        # So we can compute DP once, then test all remaining candidates against it.
        # If one succeeds, we add it, recompute DP, and repeat.
        
        S = []
        dp = [0]
        dp[0] = 1
        remaining = list(range(1, 513))
        random.shuffle(remaining)
        
        while remaining and len(S) < 48:
            if time.time() > deadline:
                break
                
            # Find a valid candidate from remaining
            valid_candidate = None
            # We might want to prioritize certain candidates?
            # Random order is fine.
            
            # Optimization: If we find one, we break and add it.
            # But maybe we should pick the "best" one? 
            # "Best" is hard to define. Random is a good heuristic for diversity.
            
            for i, x in enumerate(remaining):
                if check_validity_with_dp(dp, S, x):
                    valid_candidate = x
                    valid_index = i
                    break
            
            if valid_candidate is not None:
                S.append(valid_candidate)
                remaining.pop(valid_index)
                
                # Update DP for new S
                n = len(S)
                # Extend dp list
                dp.append(0)
                # Update existing dp entries? 
                # No, easier to recompute or update incrementally.
                # Incremental update:
                # dp_new[k] = dp_old[k] | (dp_old[k-1] << x)
                # We have dp for S_old. We need dp for S_new = S_old U {x}.
                # Let's update in place carefully.
                # We need to go from high k to low k.
                for k in range(n, 0, -1):
                    dp[k] |= (dp[k-1] << valid_candidate)
                
                # Mask? Not strictly necessary with Python ints, but keeps them smaller.
                # Max sum is n * 512. 
                # Bit length will be ~ n*512.
                
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                break
                
    return best_set
