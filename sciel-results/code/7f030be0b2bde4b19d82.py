# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Try to find a solution with 8 terms
    result = _search_k8(deadline)
    if result:
        return result
    
    # Fallback to larger k if 8 fails (should not happen if 8 is min)
    for k in range(9, 13):
        if time.time() > deadline - 5:
            break
        result = _search_k(k, deadline)
        if result:
            return result
    
    # Last resort: return a hardcoded valid solution if available, else raise
    # We don't have a hardcoded one, so we rely on the search.
    # If search fails, it's a critical error.
    raise RuntimeError("no valid representation found")

def _max_sum_float(m, start):
    """Approximate max sum of m unit fractions with denominators >= start"""
    s = 0.0
    for i in range(m):
        s += 1.0 / (start + i)
    return s

def _search_k8(deadline):
    """Search for 8 distinct denominators >= 5 summing to 1"""
    k = 8
    d = []
    
    def dfs(idx, current_sum_frac, current_sum_float, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            if current_sum_frac == 1:
                return d[:]
            return None
        
        remaining_terms = k - idx
        remaining_frac = 1 - current_sum_frac
        remaining_float = 1.0 - current_sum_float
        
        if remaining_frac <= 0:
            return None
        
        # Pruning with float for speed
        max_possible_float = _max_sum_float(remaining_terms, last_denom + 1)
        if current_sum_float + max_possible_float < 1.0 - 1e-12:
            return None
        
        # Determine range for next denominator
        # 1/d <= remaining => d >= 1/remaining
        # Use float for min_d calculation
        min_d_float = 1.0 / remaining_float
        min_d = max(last_denom + 1, int(math.ceil(min_d_float)))
        
        # Upper bound: d <= remaining_terms / remaining
        max_d_float = remaining_terms / remaining_float
        max_d = int(math.floor(max_d_float)) + 10 # small margin
        
        if min_d > max_d:
            return None
        
        # Special case: last term
        if remaining_terms == 1:
            # Need 1/d = remaining_frac
            if remaining_frac.numerator == 1:
                d_val = remaining_frac.denominator
                if d_val > last_denom and d_val <= 10000000:
                    d.append(d_val)
                    if dfs(idx + 1, current_sum_frac + Fraction(1, d_val), current_sum_float + 1.0/d_val, d_val):
                        return d[:]
                    d.pop()
            return None
        
        # Iterate possible denominators
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            if d_val <= last_denom:
                continue
            
            new_sum_float = current_sum_float + 1.0 / d_val
            if new_sum_float > 1.0:
                continue
            
            # Quick check with float
            rem_float = 1.0 - new_sum_float
            if rem_float <= 0:
                continue
            
            if remaining_terms - 1 > 0:
                max_rem_float = _max_sum_float(remaining_terms - 1, d_val + 1)
                if rem_float > max_rem_float + 1e-12:
                    continue
            
            # If float check passes, do exact fraction check
            new_sum_frac = current_sum_frac + Fraction(1, d_val)
            rem_frac = 1 - new_sum_frac
            
            # Exact check for max sum of remaining terms
            # We can skip exact max sum check if float check passed with margin
            # But to be safe, we can do it or rely on the next recursion level's check
            
            d.append(d_val)
            res = dfs(idx + 1, new_sum_frac, new_sum_float, d_val)
            if res:
                return res
            d.pop()
        
        return None
    
    return dfs(0, Fraction(0), 0.0, 4)

def _search_k(k, deadline):
    """General search for k denominators"""
    d = []
    
    def dfs(idx, current_sum_frac, current_sum_float, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            if current_sum_frac == 1:
                return d[:]
            return None
        
        remaining_terms = k - idx
        remaining_frac = 1 - current_sum_frac
        remaining_float = 1.0 - current_sum_float
        
        if remaining_frac <= 0:
            return None
        
        max_possible_float = _max_sum_float(remaining_terms, last_denom + 1)
        if current_sum_float + max_possible_float < 1.0 - 1e-12:
            return None
        
        min_d_float = 1.0 / remaining_float
        min_d = max(last_denom + 1, int(math.ceil(min_d_float)))
        max_d_float = remaining_terms / remaining_float
        max_d = int(math.floor(max_d_float)) + 10
        
        if min_d > max_d:
            return None
        
        if remaining_terms == 1:
            if remaining_frac.numerator == 1:
                d_val = remaining_frac.denominator
                if d_val > last_denom and d_val <= 10000000:
                    d.append(d_val)
                    if dfs(idx + 1, current_sum_frac + Fraction(1, d_val), current_sum_float + 1.0/d_val, d_val):
                        return d[:]
                    d.pop()
            return None
        
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            if d_val <= last_denom:
                continue
            
            new_sum_float = current_sum_float + 1.0 / d_val
            if new_sum_float > 1.0:
                continue
            
            rem_float = 1.0 - new_sum_float
            if rem_float <= 0:
                continue
            
            if remaining_terms - 1 > 0:
                max_rem_float = _max_sum_float(remaining_terms - 1, d_val + 1)
                if rem_float > max_rem_float + 1e-12:
                    continue
            
            new_sum_frac = current_sum_frac + Fraction(1, d_val)
            
            d.append(d_val)
            res = dfs(idx + 1, new_sum_frac, new_sum_float, d_val)
            if res:
                return res
            d.pop()
        
        return None
    
    return dfs(0, Fraction(0), 0.0, 4)
# EVOLVE-BLOCK-END
