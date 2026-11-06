import time
import sys

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    def time_up():
        return time.time() - start_time > deadline

    def search(k, min_d, remaining_num, remaining_den):
        """
        Find k distinct denominators >= min_d such that sum(1/d) = remaining_num/remaining_den.
        Returns list of denominators or None.
        """
        if time_up():
            return None
            
        if k == 0:
            if remaining_num == 0 and remaining_den == 1:
                return []
            return None
        
        if k == 1:
            if remaining_den % remaining_num == 0:
                d = remaining_den // remaining_num
                if d >= min_d:
                    return [d]
            return None
        
        # Pruning:
        # Max sum with k terms starting from min_d is k / min_d
        if remaining_num * min_d > remaining_den * k:
            return None
        
        # Min sum with k terms starting from min_d is k / (min_d + k - 1)
        if remaining_num * (min_d + k - 1) < remaining_den * k:
            return None

        # Determine range for next denominator d
        # 1/d <= remaining_num/remaining_den  => d >= remaining_den/remaining_num
        # Also d >= min_d
        lower = max(min_d, remaining_den // remaining_num + 1)
        
        # Upper bound for d:
        # We need k-1 terms after d.
        # Sum of next k-1 terms >= remaining_num/remaining_den - 1/d
        # Max sum of next k-1 terms is (k-1) / (d+1)
        # So remaining_num/remaining_den - 1/d <= (k-1) / (d+1)
        # (remaining_num/remaining_den - 1/d) * (d+1) <= k-1
        # This is complex. Simpler upper bound:
        # 1/d + (k-1)/(d+1) >= remaining_num/remaining_den
        # d must be small enough.
        # If 1/d is too small, we can't reach the sum even with optimal subsequent terms.
        # Actually, if d is very large, 1/d is small.
        # Let's iterate d from lower to some upper limit.
        # Upper limit: if d > remaining_den * k / remaining_num, then k/d < remaining_num/remaining_den.
        # So upper = remaining_den * k // remaining_num
        
        upper = remaining_den * k // remaining_num
        
        for d in range(lower, upper + 1):
            if time_up():
                return None
            # Check if d is valid
            # New remaining sum: remaining_num/remaining_den - 1/d = (remaining_num*d - remaining_den) / (remaining_den*d)
            new_num = remaining_num * d - remaining_den
            new_den = remaining_den * d
            if new_num < 0:
                break # d is too large, 1/d > remaining sum
            
            # Simplify fraction? Not strictly necessary for logic, but helps with integer overflow in Python (which doesn't exist) or speed.
            # We'll keep it as is to avoid GCD overhead in tight loop, but ensure we don't recurse with huge numbers.
            # For k=4,5, numbers stay manageable.
            
            res = search(k - 1, d + 1, new_num, new_den)
            if res is not None:
                return [d] + res
        
        return None

    # Try k=4
    for k in [4, 5, 6, 7, 8]:
        res = search(k, 5, 1, 1)
        if res is not None:
            # Validate
            if len(set(res)) == len(res) and all(5 <= d <= 10000000 for d in res):
                # Check sum exactly
                from fractions import Fraction
                s = sum(Fraction(1, d) for d in res)
                if s == 1:
                    return res
    raise RuntimeError("No solution found")
