import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # We know k >= 7. We try k=7, 8, ...
    # It is highly likely that a solution exists for small k.
    # Known identity for 1 with denominators >= 5?
    # 1/5 + 1/7 + 1/10 + 1/14 + 1/35 = 1? No.
    # Let's rely on the search.
    
    def search(k, last_d, current_sum, remaining_terms):
        """
        Try to find 'remaining_terms' denominators > last_d 
        that sum to 1 - current_sum.
        """
        if remaining_terms == 0:
            return current_sum == Fraction(1, 1)
        
        # Pruning 1: If current sum > 1, fail
        if current_sum > 1:
            return False
            
        # Remaining sum needed
        rem_sum = Fraction(1, 1) - current_sum
        
        # Pruning 2: Maximum possible sum with remaining_terms
        # We must pick denominators d1 < d2 < ... < dm
        # d1 > last_d
        # To maximize sum, we pick smallest possible denominators: last_d+1, last_d+2, ...
        max_possible = Fraction(0, 1)
        for i in range(remaining_terms):
            d = last_d + 1 + i
            if d > 10000000:
                return False # Cannot fit terms
            max_possible += Fraction(1, d)
            
        if max_possible < rem_sum:
            return False
            
        # Pruning 3: Minimum possible sum?
        # The smallest sum is achieved with largest denominators.
        # But we have an upper bound 10000000.
        # If rem_sum is too large to be formed by even the smallest denominators? 
        # That's covered by max_possible.
        
        # If rem_sum is so small that even 1/10000000 * remaining_terms < rem_sum?
        # Actually, if rem_sum is very small, we need large denominators.
        # The minimum contribution of one term is 1/10000000.
        # So min_possible = remaining_terms / 10000000.
        # If rem_sum < min_possible, we can't reach it? 
        # No, we can reach smaller sums by using larger denominators? 
        # Wait, denominators are capped at 10000000. So the smallest unit fraction is 1/10000000.
        # So the smallest sum of 'remaining_terms' distinct denominators <= 10000000 is 
        # 1/10000000 + 1/9999999 + ...
        # If rem_sum is smaller than this, we fail.
        min_possible = Fraction(0, 1)
        for i in range(remaining_terms):
            d = 10000000 - i
            min_possible += Fraction(1, d)
        if rem_sum < min_possible:
            return False

        # Determine range for the next denominator d
        # 1/d <= rem_sum  => d >= 1/rem_sum
        # Also d > last_d
        min_d = max(last_d + 1, math.ceil(1 / float(rem_sum)))
        
        # Also, we must be able to form the rest.
        # The largest d we can pick is constrained by the fact that we need 
        # 'remaining_terms - 1' more terms.
        # The max sum of the remaining 'remaining_terms - 1' terms 
        # using denominators > d is bounded by 
        # sum_{i=1}^{m-1} 1/(d+i)
        # We need rem_sum - 1/d <= max_sum_remaining(d)
        # This is a complex constraint to check iteratively. 
        # However, we can bound d from above.
        # If 1/d is too small, the remaining sum (rem_sum - 1/d) might be too large 
        # to be formed by the remaining terms?
        # Actually, if d is very large, 1/d is small.
        # We need rem_sum - 1/d <= sum of max possible for remaining terms.
        # But also rem_sum - 1/d must be >= min possible for remaining terms.
        
        # A simpler upper bound for d:
        # We need to fit 'remaining_terms - 1' terms after d.
        # The smallest they can be is d+1, d+2...
        # So the max sum they can provide is approx (remaining_terms-1)/d for large d?
        # Let's just iterate d upwards.
        
        # To speed up, we can check if d is too large.
        # If 1/d + max_sum(remaining_terms-1, d) < rem_sum, then d is too small?
        # No, if d is small, 1/d is large.
        # If 1/d > rem_sum, we stop (covered by min_d).
        
        # Let's iterate d from min_d upwards.
        # How high?
        # If d is very large, 1/d is small.
        # We need rem_sum - 1/d to be representable by remaining_terms-1 terms > d.
        # The max sum of remaining terms is bounded by sum_{i=1}^{m-1} 1/(d+i).
        # If rem_sum - 1/d > max_sum_remaining(d), then d is too small? 
        # Wait. If d is small, 1/d is large. rem_sum - 1/d is small.
        # If d is large, 1/d is small. rem_sum - 1/d is close to rem_sum.
        # The max sum of remaining terms decreases as d increases.
        # So if d is too large, rem_sum - 1/d might be > max_sum_remaining(d).
        
        # Let's find an upper bound for d.
        # We require: rem_sum - 1/d <= sum_{i=1}^{remaining_terms-1} 1/(d+i)
        # For large d, LHS ~ rem_sum, RHS ~ (remaining_terms-1)/d.
        # So rem_sum <= (remaining_terms-1)/d => d <= (remaining_terms-1)/rem_sum.
        
        max_d_bound = int((remaining_terms - 1) / float(rem_sum)) + 10
        # Also cap by 10000000
        max_d = min(max_d_bound, 10000000)
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None # Timeout signal
            
            new_sum = current_sum + Fraction(1, d)
            if new_sum > 1:
                break # Since d is increasing, 1/d is decreasing, but wait.
                # If new_sum > 1, then for larger d, 1/d is smaller, so new_sum might drop below 1?
                # No, we are adding terms. 
                # Ah, the loop is for the NEXT term.
                # If 1/d + current_sum > 1, then this d is too small?
                # No, if 1/d is too big, the sum exceeds 1.
                # Since d is increasing in the loop, 1/d is decreasing.
                # So if new_sum > 1 for current d, it might be <= 1 for larger d.
                # So we should NOT break. We should continue.
                # Wait, min_d was calculated as ceil(1/rem_sum).
                # So 1/min_d <= rem_sum.
                # So current_sum + 1/min_d <= current_sum + rem_sum = 1.
                # So new_sum should never exceed 1 if we start from min_d.
                # Unless floating point errors? We use Fraction, so exact.
                # So new_sum <= 1 is guaranteed.
                
            # Recursive step
            result = search(k, d, new_sum, remaining_terms - 1)
            if result:
                return [d] + result # Return the list of denominators? 
                # Wait, the function returns a list? 
                # My signature above was returning boolean. 
                # Let's modify to return the list of denominators or None.
        
        return None

    # Modify search to return list of denominators
    def search_list(k, last_d, current_sum, remaining_terms):
        if remaining_terms == 0:
            if current_sum == Fraction(1, 1):
                return []
            return None
        
        if current_sum > 1:
            return None
            
        rem_sum = Fraction(1, 1) - current_sum
        
        # Max possible sum check
        max_possible = Fraction(0, 1)
        for i in range(remaining_terms):
            d = last_d + 1 + i
            if d > 10000000:
                return None
            max_possible += Fraction(1, d)
        if max_possible < rem_sum:
            return None
            
        # Min possible sum check (denominators capped at 10^7)
        min_possible = Fraction(0, 1)
        for i in range(remaining_terms):
            d = 10000000 - i
            min_possible += Fraction(1, d)
        if rem_sum < min_possible:
            return None

        min_d = max(last_d + 1, math.ceil(1 / float(rem_sum)))
        
        # Upper bound for d
        if remaining_terms == 1:
            # 1/d = rem_sum => d = 1/rem_sum
            if rem_sum.numerator != 1:
                return None
            d = rem_sum.denominator
            if d <= last_d or d > 10000000:
                return None
            return [d]
            
        max_d_bound = int((remaining_terms - 1) / float(rem_sum)) + 10
        max_d = min(max_d_bound, 10000000)
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return "TIMEOUT"
            
            new_sum = current_sum + Fraction(1, d)
            
            # Optimization: If new_sum > 1, break? 
            # As established, min_d ensures new_sum <= 1.
            
            res = search_list(k, d, new_sum, remaining_terms - 1)
            if res == "TIMEOUT":
                return "TIMEOUT"
            if res is not None:
                return [d] + res
        
        return None

    for k in range(7, 257):
        if time.time() > deadline:
            break
        # Start search with last_d=4 (so next is >=5), sum=0, terms=k
        res = search_list(k, 4, Fraction(0, 1), k)
        if res == "TIMEOUT":
            break
        if res is not None:
            return res
            
    # Fallback: return a known valid solution if search fails or times out without finding one?
    # We must return a valid solution.
    # Let's construct a valid one quickly if the search fails.
    # 1/5 + 1/7 + 1/10 + 1/14 + 1/35 = 1?
    # 1/5 = 14/70, 1/7=10/70, 1/10=7/70, 1/14=5/70, 1/35=2/70. Sum=38/70. No.
    
    # Known solution:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + ...
    # Let's use a simple expansion if search fails.
    # But search should find k=7 or 8.
    
    # Emergency valid solution (k=10 or so)
    # 1/5 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 ...
    # This is getting complicated.
    # Let's just return a hardcoded valid solution if the search fails.
    # One such solution:
    # 1/5 + 1/7 + 1/10 + 1/14 + 1/35 = 1? No.
    
    # Let's generate one:
    # 1 = 1/5 + 4/5
    # 4/5 = 1/6 + 14/30 = 1/6 + 7/15
    # 7/15 = 1/7 + 4/15 - 1/7? No.
    # 7/15 = 1/8 + 29/120?
    
    # Hardcoded valid solution:
    # [5, 7, 10, 14, 35] -> Sum = 1/5+1/7+1/10+1/14+1/35 = (14+10+7+5+2)/70 = 38/70.
    
    # Let's use this known identity:
    # 1/5 + 1/7 + 1/10 + 1/14 + 1/35 is not 1.
    
    # How about:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # Sum is > 1.
    
    # Let's just return a list that sums to 1.
    # 1/5 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840 + 1/1680 + 1/3360 + 1/6720 + 1/13440 + 1/26880 + 1/53760 + 1/107520 + 1/215040 + 1/430080 + 1/860160 + 1/1720320 + 1/3440640 + 1/6881280 + 1/13762560 ...
    # This is the binary expansion of 1/5? No.
    
    # If search fails, we can use a simple greedy algorithm to fill the rest.
    # But search should succeed.
    
    raise RuntimeError("No solution found")
