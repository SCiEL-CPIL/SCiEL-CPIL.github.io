import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    max_terms_allowed = 256
    max_denom_limit = 10000000
    
    best_max = float('inf')
    best_solution = None

    # Initial fallback: Greedy solution (might be invalid if > 10^7, but gives structure)
    # We will rely on the DFS to find a valid one.
    # However, to ensure we return *something* valid if DFS fails or times out,
    # we should have a hardcoded valid solution or a very fast finder.
    
    # Let's try to find a solution with a specific small number of terms first (k=3,4,5)
    # as these are more likely to have smaller max denominators than very long sums 
    # which might require large denominators to fine-tune the sum.
    
    def search(remaining, min_den, current_denoms, current_max):
        nonlocal best_max, best_solution
        
        # Time check
        if time.time() > deadline:
            return
        
        # If we have a solution, check if it's better
        if remaining == 0:
            if current_max < best_max:
                best_max = current_max
                best_solution = list(current_denoms)
            return
        
        # If we've used too many terms, stop
        if len(current_denoms) >= max_terms_allowed:
            return
            
        # Pruning: If current max is already >= best_max, no point continuing
        # (unless we haven't found any solution yet, but best_max is inf then)
        if current_max >= best_max:
            return
            
        # Determine bounds for the next denominator d
        # 1. d >= min_den
        # 2. 1/d <= remaining  => d >= 1/remaining
        # 3. We have (max_terms_allowed - len(current_denoms)) terms left.
        #    Let k = terms left.
        #    The maximum sum we can add with k terms, each > d, is bounded by k/(d+1).
        #    So we need: remaining - 1/d <= k / (d+1)
        #    => (remaining - 1/d) * (d+1) <= k
        #    => remaining*d + remaining - 1 - 1/d <= k
        #    This is a lower bound on d? No.
        #    Let's iterate d and check.
        
        # Lower bound for d:
        # d_min = max(min_den, ceil(1/remaining))
        # Note: if 1/remaining is integer, d must be > 1/remaining if remaining > 1/d is not allowed?
        # Actually, if remaining == 1/d, we can pick d and finish.
        # So d_min = max(min_den, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        d_min = max(min_den, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # Upper bound for d:
        # We need remaining - 1/d > 0, so d > 1/remaining.
        # Also, we need to be able to sum the remainder with the remaining terms.
        # The smallest remainder we can make is when d is large.
        # But we are bounded by the max sum of remaining terms.
        # If d is too small, 1/d is too large, rem becomes negative (handled by d_min).
        # If d is too large, 1/d is small, rem is large.
        # Can rem be represented?
        # Max sum of k terms starting from d+1 is approx k/(d+1).
        # So we need remaining - 1/d <= k / (d+1).
        # Let k = max_terms_allowed - len(current_denoms)
        
        k_left = max_terms_allowed - len(current_denoms)
        
        # Iterate d from d_min upwards
        # We need a stopping condition.
        # If 1/d < remaining / (k_left + 1) roughly?
        # Actually, if d is very large, 1/d is very small.
        # remaining - 1/d approx remaining.
        # Max sum of k_left terms > d is approx k_left/d.
        # We need remaining <= k_left/d => d <= k_left / remaining.
        
        d_max = (k_left * remaining.denominator) // remaining.numerator
        if d_max > max_denom_limit:
            d_max = max_denom_limit
            
        # Also, d must be at least d_min
        if d_min > d_max:
            return
            
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return
            
            # Check if 1/d <= remaining
            if Fraction(1, d) > remaining:
                # This should not happen if d >= ceil(1/remaining), but just in case
                # Actually if d_min was calculated correctly, this is safe.
                # But if d_min < ceil(1/remaining), we might enter here.
                # Wait, d_min is ceil(1/remaining). So 1/d <= 1/ceil(1/rem) <= rem.
                # So this check is redundant but safe.
                pass
                
            rem_next = remaining - Fraction(1, d)
            
            if rem_next < 0:
                # Should not happen if d >= ceil(1/remaining)
                break
                
            new_max = max(current_max, d)
            
            # Prune if new_max >= best_max
            if new_max >= best_max:
                # Since d is increasing, new_max will only increase or stay same.
                # If new_max >= best_max, we can break the loop?
                # Yes, because for larger d, d >= current d, so max(current_max, d) >= new_max >= best_max.
                break
            
            # Check if rem_next can be represented by k_left - 1 terms > d
            # Max sum of (k_left - 1) terms > d is (k_left - 1) / (d + 1)
            if rem_next > Fraction(k_left - 1, d + 1):
                # Cannot represent, try next d?
                # As d increases, 1/d decreases, so rem_next increases.
                # But (k_left - 1) / (d + 1) decreases.
                # So the gap widens. We can break?
                # Let's check:
                # f(d) = rem - 1/d. g(d) = (k-1)/(d+1).
                # f'(d) = 1/d^2 > 0. g'(d) = -(k-1)/(d+1)^2 < 0.
                # So f increases, g decreases. If f(d) > g(d), then for all d' > d, f(d') > f(d) > g(d) > g(d').
                # So yes, we can break.
                break
                
            # Additional pruning: 
            # The minimum sum of (k_left - 1) distinct terms > d is 1/(d+1) + 1/(d+2) + ...
            # This is very small, so usually not a constraint for upper bound of d,
            # but if rem_next is extremely small, we might need very large denominators.
            # However, we are bounded by max_denom_limit.
            
            current_denoms.append(d)
            search(rem_next, d + 1, current_denoms, new_max)
            current_denoms.pop()

    # Start the search
    # We can bias the search towards smaller denominators by starting with min_den=1.
    # The algorithm naturally explores smaller d first.
    search(target, 1, [], 0)
    
    # If no solution found, use a hardcoded fallback if available or return empty/invalid
    # But the task says "Return 1..256 distinct denominators...".
    # If best_solution is None, we need a valid fallback.
    # Let's try a quick greedy with a twist or a known small solution.
    # Known solution for 5/31 with small max denom:
    # 5/31 = 1/7 + 1/10 + 1/230 + 1/460 is invalid (sum check failed earlier).
    # Let's check 1/8 + 1/28 + 1/1736 (from inspiration 1).
    # 1/8 + 1/28 = 7/56 + 2/56 = 9/56.
    # 5/31 - 9/56 = (280 - 279) / (31*56) = 1 / 1736.
    # So 1/8 + 1/28 + 1/1736 = 5/31.
    # Max denom is 1736.
    
    if best_solution is None:
        # Fallback valid solution
        best_solution = [8, 28, 1736]
        best_max = 1736
    else:
        # Validate the found solution
        if 0 < len(best_solution) <= 256:
            if all(1 <= d <= 10000000 for d in best_solution):
                if len(set(best_solution)) == len(best_solution):
                    s = sum(Fraction(1, d) for d in best_solution)
                    if s != target:
                        # Should not happen, but safety
                        best_solution = [8, 28, 1736]
                        best_max = 1736
                else:
                    best_solution = [8, 28, 1736]
                    best_max = 1736
            else:
                best_solution = [8, 28, 1736]
                best_max = 1736
        else:
            best_solution = [8, 28, 1736]
            best_max = 1736
            
    return best_solution
