from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline to be safe

    # Precomputed minimal-length Egyptian fraction expansion of 1
    # with all denominators >= 5.
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/210 + 1/420 = 1
    # Let's verify:
    # LCM of 5,6,7,42,140,210,420 = 420
    # 420/5 = 84, 420/6 = 70, 420/7 = 60, 420/42 = 10, 420/140 = 3, 420/210 = 2, 420/420 = 1
    # Sum = 84+70+60+10+3+2+1 = 230. Not 1.
    
    # Let me find a correct 7-term expansion.
    # Known: 1 = 1/2 + 1/3 + 1/7 + 1/43 + 1/1806 + 1/76423 + 1/... 
    # But we need denominators >= 5.
    
    # Let's try a different approach. Use the greedy algorithm or known identities.
    # Actually, let me just use a search but with a much tighter bound.
    
    # For N=7, the search space is small enough if we prune well.
    # Let's try N=7 first, then N=8, etc.
    
    def search(remaining_sum, min_den, terms_left, current_denoms):
        if time.time() > deadline:
            return None
        
        if terms_left == 0:
            if remaining_sum == 0:
                return current_denoms
            return None
        
        if terms_left == 1:
            if remaining_sum > 0:
                d = 1 / remaining_sum
                if d == int(d):
                    d_int = int(d)
                    if d_int >= min_den and d_int <= 10000000:
                        return current_denoms + [d_int]
            return None
        
        # Pruning: max possible sum
        if remaining_sum > Fraction(terms_left, min_den):
            return None
        
        # Pruning: min possible sum
        if remaining_sum < Fraction(terms_left, 10000000):
            return None
        
        # Determine valid range for d
        # 1/d < remaining_sum => d > 1/remaining_sum
        # Also d >= min_den
        lower_bound = max(min_den, int(1 / remaining_sum) + 1)
        
        # Upper bound: we need to leave enough sum for remaining terms
        # remaining_sum - 1/d >= (terms_left - 1) / 10000000
        # 1/d <= remaining_sum - (terms_left - 1) / 10000000
        # d >= 1 / (remaining_sum - (terms_left - 1) / 10000000)
        # This gives a lower bound, not upper.
        
        # For upper bound: remaining_sum - 1/d <= (terms_left - 1) / (d + 1)
        # This is complex. Let's just iterate.
        
        upper_bound = 10000000
        
        for d in range(lower_bound, upper_bound + 1):
            if time.time() > deadline:
                return None
            
            if Fraction(1, d) >= remaining_sum:
                break
            
            new_rem = remaining_sum - Fraction(1, d)
            
            if new_rem <= 0:
                continue
            
            # Max sum of next terms: (terms_left-1) / (d+1)
            if new_rem > Fraction(terms_left - 1, d + 1):
                break
            
            # Min sum of next terms: (terms_left-1) / 10000000
            if new_rem < Fraction(terms_left - 1, 10000000):
                continue
            
            res = search(new_rem, d + 1, terms_left - 1, current_denoms + [d])
            if res:
                return res
        
        return None

    # Try N=7 first
    result = search(Fraction(1, 1), 5, 7, [])
    if result:
        return result
    
    # Fallback to N=8
    result = search(Fraction(1, 1), 5, 8, [])
    if result:
        return result
    
    # Fallback to N=9
    result = search(Fraction(1, 1), 5, 9, [])
    if result:
        return result
    
    raise RuntimeError("no solution found")
