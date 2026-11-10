import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    def max_possible_sum(last_d, m):
        """Calculate the maximum sum of m distinct unit fractions 
        with denominators strictly greater than last_d.
        To maximize the sum, we pick the smallest available denominators:
        last_d+1, last_d+2, ..., last_d+m.
        """
        if m <= 0:
            return 0
        # Sum of 1/(last_d + i) for i=1 to m
        # For small m, direct sum is fast enough.
        # Since m <= 12 (initial guess), this is trivial.
        s = 0.0
        for i in range(1, m + 1):
            s += 1.0 / (last_d + i)
        return s

    def min_possible_sum(last_d, m):
        """Calculate the minimum sum of m distinct unit fractions
        with denominators in [last_d+1, 10000000].
        To minimize, pick largest denominators: 10000000, 9999999, ...
        """
        if m <= 0:
            return 0
        s = 0.0
        for i in range(m):
            s += 1.0 / (10000000 - i)
        return s

    def search(k, m, remaining, last_d, current_denoms):
        """
        Recursively search for m denominators, all > last_d,
        summing to 'remaining'.
        current_denoms: list of selected denominators so far.
        """
        if time.time() > deadline:
            return False
        
        if m == 0:
            return abs(remaining) < 1e-9

        # Pruning:
        # 1. Max possible sum with m terms starting from last_d+1
        max_sum = max_possible_sum(last_d, m)
        if remaining > max_sum + 1e-9:
            return False
        
        # 2. Min possible sum with m terms ending at 10000000
        min_sum = min_possible_sum(last_d, m)
        if remaining < min_sum - 1e-9:
            return False
        
        # 3. The next denominator d must be > last_d
        # Also, 1/d <= remaining (since all terms are positive)
        # So d >= 1/remaining
        # And d > last_d
        # And we need to leave enough for the remaining m-1 terms.
        # The remaining m-1 terms must sum to remaining - 1/d.
        # The max sum for m-1 terms after d is max_possible_sum(d, m-1).
        # So remaining - 1/d <= max_possible_sum(d, m-1)
        # => 1/d >= remaining - max_possible_sum(d, m-1)
        # This is a bit circular because max_possible_sum depends on d.
        # Simpler bound:
        # The smallest possible value for the next term is 1/10000000.
        # The largest possible value for the next term is 1/(last_d+1).
        
        # Let's iterate d from max(last_d+1, int(1/remaining)) upwards.
        # But we can tighten the lower bound for d.
        # We need remaining - 1/d >= min_possible_sum(d, m-1)
        # And remaining - 1/d <= max_possible_sum(d, m-1)
        
        # Start d at max(last_d + 1, int(1.0 / remaining) + 1)
        # Note: if remaining is very small, 1/remaining is large.
        
        d_min = max(last_d + 1, int(1.0 / remaining) + 1)
        
        # Upper bound for d:
        # We need to be able to form the rest of the sum with m-1 terms.
        # The minimum sum of the remaining m-1 terms is approx (m-1)/10000000.
        # So 1/d >= remaining - (m-1)/10000000 is not helpful for upper bound of d.
        # Instead, consider the max sum of remaining m-1 terms.
        # remaining - 1/d <= max_possible_sum(d, m-1)
        # This inequality constrains d from above?
        # Let f(d) = 1/d + max_possible_sum(d, m-1).
        # We need f(d) >= remaining.
        # f(d) is decreasing with d.
        # So we can find an upper bound for d by binary search or just iterating.
        # Given m is small, iterating is fine, but we need a good upper bound.
        
        # A simple upper bound:
        # 1/d >= remaining - max_possible_sum(last_d+1, m-1) 
        # This doesn't give an upper bound on d directly.
        
        # Let's just iterate d. To avoid infinite loops or huge ranges, 
        # we can stop if 1/d < remaining - max_possible_sum(d, m-1) is impossible?
        # Actually, if d is too large, 1/d becomes small, and the max sum of the rest 
        # (which also decreases as d increases) becomes small.
        # So there is a cutoff.
        
        # Let's set a reasonable upper bound.
        # If m=1, d = 1/remaining exactly.
        # If m>1, d cannot be too large.
        # 1/d + (m-1)/(d+1) >= remaining is a loose upper bound check.
        # Actually, max_possible_sum(d, m-1) <= (m-1)/(d+1).
        # So 1/d + (m-1)/(d+1) >= remaining.
        # Let's solve for d roughly.
        # For now, let's just iterate and break if time runs out.
        # But we need a finite range.
        # If d is such that 1/d < remaining - (m-1)/10000000, then even with max possible rest, we can't reach remaining?
        # No, min rest is (m-1)/10000000.
        # So we need 1/d + (m-1)/10000000 >= remaining => 1/d >= remaining - (m-1)/10000000.
        # This gives a LOWER bound on 1/d, hence an UPPER bound on d.
        # d_max = int(1.0 / (remaining - (m-1)/10000000)) if remaining > (m-1)/10000000 else 10000000
        
        denom_remaining_min = (m - 1) / 10000000.0
        if remaining <= denom_remaining_min:
            return False # Can't form remaining sum even with max denominators
        
        d_max = int(1.0 / (remaining - denom_remaining_min))
        if d_max > 10000000:
            d_max = 10000000
            
        # Also, d must be > last_d
        if d_min > d_max:
            return False
            
        # Iterate d from d_min to d_max
        # Optimization: step size? No, we need exact integers.
        # But d_max can be large.
        # However, the constraint 1/d + max_possible_sum(d, m-1) >= remaining is stronger.
        # Let's use it to prune d_max further.
        # f(d) = 1/d + sum_{j=1}^{m-1} 1/(d+j)
        # We need f(d) >= remaining.
        # Since f(d) is decreasing, we can find the largest d satisfying this.
        # For small m, we can just check in the loop.
        
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return False
                
            # Check if remaining - 1/d can be formed by m-1 terms > d
            new_remaining = remaining - 1.0 / d
            if new_remaining < 0:
                continue
                
            # Pruning: max sum for m-1 terms starting from d+1
            if new_remaining > max_possible_sum(d, m - 1) + 1e-9:
                continue
                
            # Pruning: min sum for m-1 terms ending at 10^7
            if new_remaining < min_possible_sum(d, m - 1) - 1e-9:
                # Since min sum decreases as d increases (denoms get larger? No, min sum is fixed by max denom 10^7)
                # Wait, min_possible_sum(d, m-1) uses denominators 10^7, 10^7-1, ...
                # It doesn't depend on d, except that d must be < those denominators.
                # If d >= 10^7 - (m-2), we can't pick m-1 distinct denominators > d in range.
                # But our range is up to 10^7.
                # If d is close to 10^7, min sum is not achievable if not enough room.
                # However, min_possible_sum assumes we CAN pick them.
                # If d is too large, we might not have enough distinct integers > d up to 10^7.
                # We need 10^7 - d >= m - 1.
                if 10000000 - d < m - 1:
                    break # No more valid d
                    
            # Recursive call
            current_denoms.append(d)
            if search(k, m - 1, new_remaining, d, current_denoms):
                return True
            current_denoms.pop()
            
        return False

    # Try k from 12 upwards
    # Known solutions exist for small k. 
    # For k=12, a known solution is:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/13 + 1/14 + 1/15 + 1/18 + 1/20 ? No, sum must be 1.
    # Let's just run the search.
    
    for k in range(12, 257):
        # Initial call: m=k, remaining=1.0, last_d=4 (since d>=5)
        if search(k, k, 1.0, 4, []):
            return construct_result()
            
    raise RuntimeError("No solution found within time limit or k range")

# Helper to format result, though construct returns the list directly
# The prompt says: "Return 1..256 distinct integer denominators"
# The current program returns a list.
# I will modify the search to store the result.

# Refactoring to store result globally or return it properly.
result = []

def construct():
    global result
    start_time = time.time()
    deadline = start_time + 55.0
    
    def max_possible_sum(last_d, m):
        if m <= 0:
            return 0
        s = 0.0
        for i in range(1, m + 1):
            s += 1.0 / (last_d + i)
        return s

    def min_possible_sum(last_d, m):
        if m <= 0:
            return 0
        s = 0.0
        for i in range(m):
            s += 1.0 / (10000000 - i)
        return s

    def search(m, remaining, last_d, current_denoms):
        if time.time() > deadline:
            return False
        
        if m == 0:
            if abs(remaining) < 1e-9:
                result.extend(current_denoms)
                return True
            return False

        max_sum = max_possible_sum(last_d, m)
        if remaining > max_sum + 1e-9:
            return False
        
        min_sum = min_possible_sum(last_d, m)
        if remaining < min_sum - 1e-9:
            return False
        
        d_min = max(last_d + 1, int(1.0 / remaining) + 1)
        
        denom_remaining_min = (m - 1) / 10000000.0
        if remaining <= denom_remaining_min:
            return False
        
        d_max = int(1.0 / (remaining - denom_remaining_min))
        if d_max > 10000000:
            d_max = 10000000
            
        if d_min > d_max:
            return False
            
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return False
                
            if 10000000 - d < m - 1:
                break
                
            new_remaining = remaining - 1.0 / d
            if new_remaining < 0:
                continue
                
            if new_remaining > max_possible_sum(d, m - 1) + 1e-9:
                continue
                
            if new_remaining < min_possible_sum(d, m - 1) - 1e-9:
                continue

            current_denoms.append(d)
            if search(m - 1, new_remaining, d, current_denoms):
                return True
            current_denoms.pop()
            
        return False

    for k in range(12, 257):
        result = []
        if search(k, 1.0, 4, []):
            return result
            
    raise RuntimeError("No solution found")

