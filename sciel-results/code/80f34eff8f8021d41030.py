"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def _is_valid_answer(ans, divisors):
    """Check if answer is valid: 10 distinct positive ints <= 1e6, each divisible by corresponding divisor."""
    if not isinstance(ans, (list, tuple)):
        return False
    if len(ans) != len(divisors):
        return False
    seen = set()
    for i, (val, div) in enumerate(zip(ans, divisors)):
        if not isinstance(val, int) or isinstance(val, bool):
            return False
        if val < 1 or val > P['maximum']:
            return False
        if val % div != 0:
            return False
        if val in seen:
            return False
        seen.add(val)
    return True

def _span(ans):
    return max(ans) - min(ans)

def construct():
    """
    Find 10 distinct positive integers <= 1000000, aligned with divisors [6,10,14,15,21,22,26,33,35,39].
    Each representative must be divisible by its corresponding input.
    Minimize max(representatives) - min(representatives).
    """
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Fallback: The divisors themselves are always a valid solution.
    # Span is 39 - 6 = 33.
    best_ans = divisors[:]
    best_span = _span(best_ans)
    
    # We search for the minimum span S.
    # Lower bound: 9 (10 distinct integers).
    # Upper bound: 33 (from fallback).
    # We iterate S from 9 to 33.
    
    # To optimize the inner loop, we precompute nothing heavy, but we 
    # will check windows [L, L+S] for L in a reasonable range.
    # The optimal L is likely small. We check L from 1 to 5000.
    # If not found, we might need to go higher, but 5000 is usually enough 
    # for such small divisors.
    
    # We can also be smarter: 
    # For a fixed S, a valid window must contain a multiple of the largest divisor (39).
    # So we can iterate over multiples of 39, say M, and check L in [M-S, M].
    # This reduces the number of L checks significantly.
    
    for S in range(9, best_span + 1):
        if time.time() > deadline:
            break
            
        found_solution = False
        
        # Iterate over potential "anchors" which are multiples of the largest divisor (39)
        # This ensures we don't miss any window that could potentially fit, 
        # because any valid window of small span S (< 39) must contain 
        # at least one multiple of 39? 
        # Wait, if S < 39, a window might NOT contain a multiple of 39.
        # Example: Window [1, 10], S=9. Multiples of 39: 39, 78... None in [1,10].
        # So this anchor strategy is INVALID for S < 39.
        
        # Correct Strategy:
        # Just iterate L from 1 to a limit.
        # Limit: Since the pattern repeats with LCM, and LCM is large, 
        # but for small S, the "phase" of the window matters.
        # Checking L=1..5000 is a heuristic. 
        # To be more rigorous, we can check L=1..max(divisors)*10.
        # Let's use 2000 as a safe limit for small S.
        
        limit_L = 2000
        
        for L in range(1, limit_L + 1):
            if time.time() > deadline:
                break
                
            if L + S > max_val:
                continue
                
            # Check feasibility for window [L, L+S]
            # For each divisor i, find valid multiples in [L, L+S]
            options = []
            valid = True
            
            for i, d in enumerate(divisors):
                # Smallest multiple >= L
                k_min = (L + d - 1) // d
                # Largest multiple <= L+S
                k_max = (L + S) // d
                
                if k_min > k_max:
                    valid = False
                    break
                    
                # Generate list of values
                vals = [k * d for k in range(k_min, k_max + 1)]
                options.append((i, d, vals))
            
            if not valid:
                continue
                
            # Sort options by length (most constrained first) to speed up backtracking
            options.sort(key=lambda x: len(x[2]))
            
            # Backtracking search
            assigned = [None] * n
            used_values = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                orig_idx, d, vals = options[idx]
                for v in vals:
                    if v not in used_values:
                        used_values.add(v)
                        assigned[orig_idx] = v
                        if backtrack(idx + 1):
                            return True
                        used_values.remove(v)
                        assigned[orig_idx] = None
                return False
            
            if backtrack(0):
                # Found a valid solution
                ans = [int(v) for v in assigned]
                if _is_valid_answer(ans, divisors):
                    # Since we iterate S from low to high, the first one found is optimal
                    return ans
    
    # If no better solution found in the heuristic range, return the fallback
    return [int(v) for v in best_ans]

# EVOLVE-BLOCK-END
