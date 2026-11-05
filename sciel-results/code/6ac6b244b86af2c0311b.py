"""Rewritten program for the suite task; finds minimal Egyptian fraction for 1."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    timeout = 55.0  # Keep a safety margin
    
    # Target is 1.
    # We search for a set of distinct integers > 4 summing to 1.
    # We try to minimize the number of terms.
    
    def search(start_denom, remaining, max_terms, current):
        # remaining: numerator/denominator fraction remaining to sum to 1
        # max_terms: number of terms we are aiming for
        # current: list of chosen denominators
        if max_terms == 0:
            return None
        
        # Check time limit
        if time.time() - start_time > timeout:
            return None
            
        # Pruning: if the maximum possible sum with 'max_terms' terms starting from 'start_denom'
        # is less than 'remaining', we can't reach the target.
        # Max sum approx sum of 1/start_denom, 1/(start_denom+1), ...
        # A tighter bound: if 1/start_denom * max_terms < remaining, return None?
        # Actually, the sum of reciprocals diverges, but for small terms it's bounded.
        # Simple bound: if remaining > max_terms / start_denom, return None.
        
        if remaining * start_denom > max_terms:
            return None
            
        # Try denominators from start_denom upwards
        # Upper bound for denom: remaining * denom <= 1 => denom <= 1/remaining
        # Also distinct and > 4
        max_d = int(1 / remaining)
        if max_d < start_denom:
            return None
            
        # We should not exceed a reasonable max denominator
        # If we are close to the end, we can be more precise
        for d in range(start_denom, max_d + 1):
            if time.time() - start_time > timeout:
                return None
                
            # new_remaining = remaining - 1/d
            # Using fractions for exactness
            # remaining = num/den
            # new_remaining = num/den - 1/d = (num*d - den) / (den*d)
            num, den = remaining
            new_num = num * d - den
            new_den = den * d
            
            if new_num < 0:
                break # Sum exceeded 1
            
            if new_num == 0:
                if max_terms == 1:
                    return current + [d]
                # If we hit 0 with terms left, we need to split 0? No, sum is exactly 1.
                # We must use exactly 'max_terms' terms?
                # The task says "Return 1..256 distinct... minimize their count".
                # So we can stop if we hit 1.
                return current + [d]
                
            # Reduce fraction
            # Simple GCD
            g = 1
            for i in range(2, int(min(abs(new_num), new_den) ** 0.5) + 1):
                if new_num % i == 0 and new_den % i == 0:
                    g = i
            # This GCD loop is slow. Use math.gcd
            import math
            g = math.gcd(abs(new_num), new_den)
            new_num //= g
            new_den //= g
            
            if new_num == 0:
                return current + [d]
                
            # Recurse
            res = search(d + 1, (new_num, new_den), max_terms - 1, current + [d])
            if res:
                return res
                
        return None

    # Try k=8, then 9, etc.
    for k in range(8, 15):
        # Initial remaining is 1/1
        res = search(5, (1, 1), k, [])
        if res:
            return res
            
    # Fallback: return a known valid set of size 10 if search fails
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 ?
    # Sum is approx 1.05.
    # Let's just return the baseline's DP result logic but optimized?
    # Or a hardcoded set.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 ...
    
    # A valid set:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20
    # This sum is > 1.
    
    # Let's return a set that we know sums to 1.
    # 1 = 1/2 + 1/3 + 1/6
    # Split 1/2 -> 1/5 + 1/5 (no)
    # Split 1/2 -> 1/6 + 1/3 (no)
    
    # Use the DP from the baseline as a fallback
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
