import random

def construct():
    max_n = 512
    max_size = 48
    
    def get_violation_counts(S):
        """
        Returns a list of tuples (element, count) where count is the number of violations
        where the element is the mean.
        Also returns the total number of violations.
        """
        counts = {x: 0 for x in S}
        total_violations = 0
        S_list = list(S)
        n = len(S_list)
        
        # For each element a, check if it is a mean of any subset of S \ {a}
        for a in S_list:
            # Build DP for S \ {a}
            # dp[k] is a bitmask of sums achievable with k elements
            max_sum = sum(y for y in S_list if y != a)
            # We only need sums up to k*a, but k varies.
            # Max sum we care about for size k is k*a.
            # Max k is n-1. Max sum is (n-1)*a.
            # But we can just compute up to max_sum.
            
            # Optimization: Only check k such that k*a <= max_sum
            # And k >= 2
            
            dp = [0] * (n) # max size n-1
            dp[0] = 1
            
            for y in S_list:
                if y == a:
                    continue
                # Update dp in reverse
                # We need to track size.
                # Current max size is number of elements processed so far.
                # Let's just iterate k from current_max down to 1.
                # To be safe, iterate n-1 down to 1.
                for k in range(n - 1, 0, -1):
                    # Shift dp[k-1] by y
                    # We must mask to avoid overflow or keep it small?
                    # Python ints handle arbitrary size.
                    dp[k] |= (dp[k - 1] << y)
            
            # Now check for violations
            # A violation exists if for some k >= 2, bit (k*a) is set in dp[k]
            for k in range(2, n):
                target_sum = k * a
                if target_sum > max_sum:
                    break
                if (dp[k] >> target_sum) & 1:
                    counts[a] += 1
                    total_violations += 1
        return counts, total_violations

    # Initial set: 1..48
    S = list(range(1, 49))
    
    # Greedy Removal
    # We want to reach 0 violations.
    # If we have > 48 elements, we remove until 48? No, we start with 48.
    # We remove elements to fix violations. This reduces size.
    # Then we try to add elements back to reach 48.
    
    for _ in range(100): # Iterations
        counts, total_violations = get_violation_counts(S)
        if total_violations == 0:
            break
        
        # Find the element with the max count
        max_count = -1
        worst_elem = None
        for elem, count in counts.items():
            if count > max_count:
                max_count = count
                worst_elem = elem
        
        if worst_elem is None:
            break
            
        S.remove(worst_elem)
        
        # If size is too small, we might need to add back later.
        # But first, let's get to a valid set.

    # Now S is valid but might be smaller than 48.
    # Try to add elements from 1..512 to increase size.
    candidates = [x for x in range(1, max_n + 1) if x not in S]
    random.shuffle(candidates)
    
    for x in candidates:
        if len(S) >= max_size:
            break
        # Check if x can be added
        # We need to check if x is a mean of S, and if adding x makes any a in S a mean of a new subset.
        
        # Simple check: Just try adding and see if violations increase.
        # But that's expensive.
        # Let's use the previous logic:
        # 1. Check if x is mean of S.
        # 2. Check if any a in S becomes mean of subset including x.
        
        # To be safe and fast, we can just use get_violation_counts on S U {x}
        # and see if it's 0.
        # But that's O(N^3) per candidate.
        
        # Better:
        # Check 1: x is mean of S?
        # Build DP for S.
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1
        for y in S:
            for k in range(n, 0, -1):
                dp[k] |= (dp[k - 1] << y)
        
        x_is_mean = False
        for k in range(2, n + 1):
            if (dp[k] >> (k * x)) & 1:
                x_is_mean = True
                break
        
        if x_is_mean:
            continue
            
        # Check 2: Any a in S becomes mean?
        # a is mean of T U {x} where T subset S \ {a}.
        # sum(T) + x = |T| a => sum(T) = |T| a - x.
        
        valid = True
        for a in S:
            # Check if S \ {a} has subset of size j summing to j*a - x
            # We can reuse the DP logic or do a quick check.
            # Since we are adding one element, we can just check against the existing DP of S?
            # No, T is subset of S \ {a}.
            
            # Let's just do the full check for this a.
            # It's expensive but we only do it for valid x.
            
            # Actually, we can just call get_violation_counts(S + [x])
            # and see if total_violations == 0.
            # If we do this for 48 candidates, it's 48 * 0.1s = 4.8s.
            # Acceptable.
            
            new_S = S + [x]
            _, violations = get_violation_counts(new_S)
            if violations == 0:
                S = new_S
                break
            else:
                valid = False
                break # This x caused a violation, try next x? 
                # Wait, if x caused a violation, we should skip x.
                # But the violation might be due to a different a.
                # So if violations > 0, x is not a good addition.
        
    return S
