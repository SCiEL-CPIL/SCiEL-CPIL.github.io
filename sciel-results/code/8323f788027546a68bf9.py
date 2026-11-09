import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    def can_form_mean(a, others):
        """Check if a is the mean of any subset of others with size >= 2."""
        n = len(others)
        if n < 2:
            return False
        # differences
        diffs = [y - a for y in others]
        # We need subset of size >= 2 with sum 0
        # DP: for each possible sum, track the min and max subset sizes that achieve it
        # Sums range from -sum(abs(diffs)) to sum(abs(diffs))
        # Use a dict: sum -> (min_size, max_size) or just track achievable (sum, size) pairs
        
        # Better: use a set of (sum, size) pairs, but that can explode.
        # Alternative: for each size k from 2 to n, check if sum 0 is achievable with exactly k elements.
        # Use bitset DP: dp[k] is a set of achievable sums with exactly k elements.
        # But sums can be large in range.
        
        # Since n <= 47 and diffs are bounded, let's use:
        # dp as list of sets, dp[k] = set of sums achievable with k elements
        # Start: dp[0] = {0}
        # For each diff, update dp
        
        max_sum = sum(abs(d) for d in diffs)
        # dp[k] = set of achievable sums with exactly k elements
        dp = [set() for _ in range(n + 1)]
        dp[0].add(0)
        
        for d in diffs:
            # iterate k in reverse to avoid using same element twice
            for k in range(n - 1, -1, -1):
                if not dp[k]:
                    continue
                new_sums = set()
                for s in dp[k]:
                    new_sums.add(s + d)
                dp[k + 1] = dp[k + 1] | new_sums
            # Early exit: check if 0 is in dp[k] for any k >= 2
            # Actually check after processing all, or check incrementally
            for k in range(2, n + 1):
                if 0 in dp[k]:
                    return True
        
        return False
    
    # Greedy construction with fast validity check
    selected = []
    for x in range(1, 513):
        if len(selected) >= 48:
            break
        if time.time() > deadline - 10:
            break
        
        candidate = selected + [x]
        valid = True
        # Check all elements in candidate
        for i, a in enumerate(candidate):
            others = [candidate[j] for j in range(len(candidate)) if j != i]
            if can_form_mean(a, others):
                valid = False
                break
        
        if valid:
            selected = candidate
    
    # Ensure we return exactly min(48, len(selected)) elements
    return selected[:48]
