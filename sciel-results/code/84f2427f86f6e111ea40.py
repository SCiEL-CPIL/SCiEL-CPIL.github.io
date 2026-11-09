import math

def construct():
    target = math.factorial(12)
    
    # Generate all divisors of target
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
            
    # Filter divisors > 12
    valid_divs = [d for d in divisors if d > 12]
    valid_divs.sort()
    
    # We established that max number of factors is 7.
    MAX_FACTORS = 7
    
    # Precompute suffix products for pruning?
    # Or just calculate on the fly in DFS?
    # Given depth <= 7, on-the-fly product of up to 7 numbers is fast.
    
    def dfs(start_idx, remaining, count):
        if remaining == 1:
            return 1 <= count <= MAX_FACTORS
        
        if count >= MAX_FACTORS:
            return False
            
        if start_idx >= len(valid_divs):
            return False
            
        # Pruning: Max possible product with remaining slots
        slots_left = MAX_FACTORS - count
        # We can pick at most 'slots_left' factors from valid_divs[start_idx:]
        # To get the max product, we pick the largest available ones.
        # However, we are constrained by 'remaining'.
        # If the product of the largest 'slots_left' divisors in the suffix is < remaining, 
        # it's impossible.
        
        # Get the largest 'slots_left' divisors from valid_divs[start_idx:]
        # Note: valid_divs is sorted ascending.
        # The largest are at the end of the list.
        # But we are limited by 'start_idx'.
        # So we look at valid_divs[start_idx:].
        # The largest elements are valid_divs[-1], valid_divs[-2], etc.
        # But they must be >= valid_divs[start_idx].
        # Since the list is sorted, valid_divs[start_idx:] contains all elements from start_idx to end.
        # The largest 'slots_left' elements in this subarray are simply the last 'slots_left' elements of the whole array?
        # No, they are the last 'slots_left' elements of the subarray valid_divs[start_idx:].
        # Which are valid_divs[-1], ..., valid_divs[-slots_left] IF they are >= valid_divs[start_idx].
        # Since start_idx increases, the pool shrinks from the left.
        # The right end is always the same (the largest divisors).
        # So the max product is the product of the largest min(slots_left, len(valid_divs)-start_idx) elements in valid_divs.
        
        available = valid_divs[start_idx:]
        if len(available) == 0:
            return False
            
        # Take the largest k elements from available
        k = min(slots_left, len(available))
        # The largest k elements are available[-k:]
        # Calculate their product
        max_prod = 1
        for j in range(len(available) - k, len(available)):
            max_prod *= available[j]
            
        if max_prod < remaining:
            return False
            
        # Additional Pruning:
        # If remaining is prime and > max(available), fail.
        # Actually, if remaining has a prime factor p such that p > max(available), 
        # and p != remaining (i.e. remaining is composite), it might still be possible?
        # No, if p is a factor of remaining, p must be formed by factors from available.
        # The smallest factor in available is available[0].
        # If p < available[0], it cannot be formed.
        # But p is a prime factor of remaining.
        # This check is complex. The max_prod check is usually sufficient for this size.
        
        for i in range(start_idx, len(valid_divs)):
            d = valid_divs[i]
            if d > remaining:
                break
            if remaining % d == 0:
                if dfs(i + 1, remaining // d, count + 1):
                    return True
        return False

    # Iterate limit M
    # We need to return the factors, not just True/False.
    # So we need to modify DFS to return the list of factors.
    
    def dfs_factors(start_idx, remaining, count, current_factors):
        if remaining == 1:
            if 1 <= count <= MAX_FACTORS:
                return current_factors
            return None
        
        if count >= MAX_FACTORS:
            return None
            
        if start_idx >= len(valid_divs):
            return None
            
        slots_left = MAX_FACTORS - count
        available = valid_divs[start_idx:]
        if len(available) == 0:
            return None
            
        k = min(slots_left, len(available))
        max_prod = 1
        for j in range(len(available) - k, len(available)):
            max_prod *= available[j]
            
        if max_prod < remaining:
            return None
            
        for i in range(start_idx, len(valid_divs)):
            d = valid_divs[i]
            if d > remaining:
                break
            if remaining % d == 0:
                res = dfs_factors(i + 1, remaining // d, count + 1, current_factors + [d])
                if res is not None:
                    return res
        return None

    # We want to minimize the largest factor.
    # So we iterate M from smallest to largest.
    # For each M, we restrict the pool to divisors <= M.
    # Instead of slicing valid_divs, we can pass a max_idx to DFS.
    
    def dfs_factors_limited(start_idx, remaining, count, current_factors, max_idx):
        if remaining == 1:
            if 1 <= count <= MAX_FACTORS:
                return current_factors
            return None
        
        if count >= MAX_FACTORS:
            return None
            
        if start_idx > max_idx:
            return None
            
        slots_left = MAX_FACTORS - count
        # Available divisors are valid_divs[start_idx : max_idx+1]
        # Length of available: max_idx - start_idx + 1
        avail_len = max_idx - start_idx + 1
        if avail_len <= 0:
            return None
            
        k = min(slots_left, avail_len)
        # The largest k elements in valid_divs[start_idx : max_idx+1] are at the end of this slice.
        # i.e., indices max_idx - k + 1 to max_idx.
        max_prod = 1
        for j in range(max_idx - k + 1, max_idx + 1):
            max_prod *= valid_divs[j]
            
        if max_prod < remaining:
            return None
            
        for i in range(start_idx, max_idx + 1):
            d = valid_divs[i]
            if d > remaining:
                break
            if remaining % d == 0:
                res = dfs_factors_limited(i + 1, remaining // d, count + 1, current_factors + [d], max_idx)
                if res is not None:
                    return res
        return None

    # Find the index of the current limit in valid_divs
    for idx, limit in enumerate(valid_divs):
        res = dfs_factors_limited(0, target, 0, [], idx)
        if res is not None:
            return res
            
    return [target]
