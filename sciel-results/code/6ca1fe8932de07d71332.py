"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill
    
    target = Fraction(5, 31)
    best_solution = [8, 28, 1736]  # Known baseline
    best_max_den = 1736
    
    def get_divisors(n):
        """Get all divisors of n"""
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs

    def try_complete_algebraic(prefix, used_set):
        """
        Try to complete the expansion given a prefix of denominators.
        We assume we want to complete with exactly 2 more terms x, y.
        Returns a list of [x, y] if found, else None.
        """
        remainder = target
        for d in prefix:
            remainder -= Fraction(1, d)
        
        if remainder < 0:
            return None
        if remainder == 0:
            # Should not happen if we expect 2 more terms, but handle it
            return []
        
        p = remainder.numerator
        q = remainder.denominator
        
        # We need 1/x + 1/y = p/q
        # (px - q)(py - q) = q^2
        # Let A = px - q, B = py - q. AB = q^2.
        # x = (A + q) / p, y = (B + q) / p
        
        divs = get_divisors(q * q)
        
        # We want to minimize the max denominator, so we might prefer smaller x, y?
        # Actually, we just need ANY valid x, y. The global best is tracked by max_den.
        # To find the one with smallest max(x,y), we should check all divisors.
        
        candidates = []
        for A in divs:
            B = (q * q) // A
            
            # x = (A + q) / p
            num_x = A + q
            if num_x % p != 0:
                continue
            x = num_x // p
            
            # y = (B + q) / p
            num_y = B + q
            if num_y % p != 0:
                continue
            y = num_y // p
            
            # Check constraints
            if x <= 0 or y <= 0:
                continue
            if x > 10000000 or y > 10000000:
                continue
            if x in used_set or y in used_set:
                continue
            if x == y:
                continue
            
            # Valid pair found
            candidates.append((x, y))
            
        if not candidates:
            return None
        
        # Return the pair that minimizes the max denominator locally?
        # Or just return the first one?
        # Since we are iterating prefixes, and we track global best, any valid pair is good.
        # However, to be efficient, we can return the pair with smallest max(x,y)
        # to potentially prune earlier in deeper searches? 
        # Actually, we are doing fixed-depth completion (last 2 terms).
        # So we just return one valid pair.
        # Let's pick the one with smallest max(x,y) to be safe.
        best_pair = min(candidates, key=lambda t: max(t))
        return [best_pair[0], best_pair[1]]

    # Search Strategy:
    # Iterate over possible prefixes of length L.
    # For each prefix, try to complete with 2 terms.
    # This covers expansions of length L+2.
    
    # We can also try completing with 1 term (if remainder is 1/d).
    # Or 3 terms?
    # The algebraic trick works for the LAST TWO terms.
    # So if we fix a prefix of length K, we are finding a solution of length K+2.
    
    # To find the global minimum max_den, we should explore small max_den solutions.
    # Solutions with small max_den likely have small denominators overall.
    
    # Let's iterate by the maximum denominator bound M.
    # But that's hard to invert.
    
    # Alternative: Iterate over prefixes with small denominators.
    # If the max denominator is small, all denominators are small.
    
    # Let's try prefixes of length 1, 2, 3, ...
    # For each prefix, ensure denominators are distinct and < current best_max_den.
    
    # We'll do a DFS for the prefix, but the completion is algebraic.
    
    def dfs_prefix(current_prefix, used_set, min_denom):
        nonlocal best_solution, best_max_den
        
        if time.time() - start_time > time_limit:
            return
        
        # Try to complete with 2 more terms
        completion = try_complete_algebraic(current_prefix, used_set)
        if completion is not None:
            full_solution = current_prefix + completion
            if len(set(full_solution)) == len(full_solution):
                mx = max(full_solution)
                if mx < best_max_den:
                    best_max_den = mx
                    best_solution = full_solution[:]
        
        # If we have a solution, we might still find a better one with more terms?
        # No, adding terms usually increases the max denominator or keeps it same?
        # Actually, splitting 1/d into 1/(d+1) + 1/(d(d+1)) increases the max denominator.
        # So fewer terms with smaller denominators is better.
        # But we are not restricted to minimal terms.
        # However, the algebraic completion finds the "tightest" 2-term completion for a given prefix.
        
        # Recurse to add one more term to the prefix
        # We need to choose the next denominator d.
        # d must be > min_denom and < best_max_den (to improve).
        # Also 1/d < remainder.
        
        remainder = target
        for d in current_prefix:
            remainder -= Fraction(1, d)
            
        if remainder <= 0:
            return
        
        # Next denominator d must satisfy:
        # 1. d > min_denom
        # 2. 1/d < remainder  => d > 1/remainder
        # 3. d < best_max_den (to find a better solution)
        
        d_min = min_denom + 1
        d_from_remainder = (remainder.denominator + remainder.numerator - 1) // remainder.numerator
        if d_from_remainder < d_min:
            d_min = d_from_remainder
        
        # Upper bound for d:
        # We need to leave enough for at least 2 more terms?
        # If we add d, we need to complete with 2 terms.
        # The smallest possible sum of 2 distinct terms > d is 1/(d+1) + 1/(d+2).
        # So remainder - 1/d >= 1/(d+1) + 1/(d+2) is not required, just possible.
        # But 1/d must be less than remainder.
        
        # Also, we shouldn't go too high.
        # If d is close to best_max_den, it's unlikely to improve.
        d_max = best_max_den - 1
        
        if d_min > d_max:
            return
            
        # Limit search depth to avoid explosion?
        # The prefix length is small.
        
        for d in range(d_min, d_max + 1):
            if time.time() - start_time > time_limit:
                return
            
            new_used = used_set | {d}
            new_prefix = current_prefix + [d]
            
            # Check if remainder - 1/d is positive
            rem_after_d = remainder - Fraction(1, d)
            if rem_after_d < 0:
                continue
            
            # If rem_after_d == 0, we have a solution of length len(prefix)+1
            # But our algebraic completion expects 2 terms.
            # If rem_after_d == 0, we can just record it.
            if rem_after_d == 0:
                full_solution = new_prefix
                if len(set(full_solution)) == len(full_solution):
                    mx = max(full_solution)
                    if mx < best_max_den:
                        best_max_den = mx
                        best_solution = full_solution[:]
                continue
            
            # Recurse
            dfs_prefix(new_prefix, new_used, d)
            
            # Heuristic: If we found a solution with this d, 
            # do we need to check larger d?
            # Larger d means smaller 1/d, larger remainder for the rest.
            # Larger remainder for 2 terms might force larger denominators.
            # So smaller d is generally better for keeping max_den small?
            # Not necessarily.
            # But we are iterating all d anyway.

    # Start with empty prefix
    dfs_prefix([], set(), 1)
    
    # If no better solution found, return baseline
    return best_solution

# EVOLVE-BLOCK-END
