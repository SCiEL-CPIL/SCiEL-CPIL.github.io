import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_VAL = 1000000
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin
    
    # Precompute: for each divisor, the list of multiples up to MAX_VAL
    # But that's too much memory. Instead, generate on the fly.
    
    # Function to get multiples of d in [lo, hi]
    def multiples_in_range(d, lo, hi):
        if lo > hi:
            return []
        first = ((lo + d - 1) // d) * d
        if first > hi:
            return []
        # Generate list
        result = []
        x = first
        while x <= hi:
            result.append(x)
            x += d
        return result
    
    # Backtracking to find distinct representatives
    def find_assignment(candidates):
        """
        candidates: list of lists, candidates[i] is the list of candidate values for divisor i
        Returns a list of chosen values or None
        """
        n = len(candidates)
        # Sort indices by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        chosen = [None] * n
        used = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    chosen[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    chosen[i] = None
            return False
        
        if backtrack(0):
            return chosen
        return None
    
    # We'll search over window width W and starting position L
    # L should be a multiple of one of the divisors
    # Generate candidate L values: multiples of each divisor up to MAX_VAL
    # But that's too many. Instead, iterate L in a smarter way.
    
    # Alternative: iterate over possible min values. The min must be a multiple of some divisor.
    # Let's collect all candidate min values (multiples of divisors) and sort them.
    # But there are ~700K of them. Instead, let's try a range of W values.
    
    best_result = None
    best_width = float('inf')
    
    # Try window widths from small to large
    # Minimum possible width: at least max(divisors) - 1? No, we need distinct values.
    # Let's start W from 10 and go up
    # But we need to try different L positions.
    
    # For efficiency, for each W, try L values that are multiples of the divisors
    # in a limited range.
    
    # Let's try W from 20 to 200 (guessing the answer is small)
    # For each W, try L from 1 to MAX_VAL - W, but only multiples of divisors.
    
    # Better: for each divisor d_i, L must be such that there's a multiple of d_i in [L, L+W]
    # and L should be close to a multiple.
    
    # Let me try a different approach: for each pair of divisors (i, j), 
    # the min is a multiple of d_i and max is a multiple of d_j.
    # This is getting complex.
    
    # Simple approach: iterate L over a grid of candidate values.
    # Candidate L values: for each divisor d, multiples of d.
    # But to limit, let's try L in steps.
    
    # Let's try W from 10 to 500, and for each W, try L at multiples of each divisor
    # in a limited range.
    
    for W in range(10, 501):
        if time.time() - start_time > DEADLINE:
            break
        # For this W, try various L
        # L should be such that for each divisor, there's at least one multiple in [L, L+W]
        # Let's try L values that are multiples of the smallest divisor (6) 
        # and also try L = 1, and L near multiples of other divisors.
        
        # Generate candidate L values: multiples of each divisor in a reasonable range
        # To limit, let's try L from 1 to 10000 first, then expand if needed.
        # Actually, the optimal L could be anywhere. Let's try a sampling approach.
        
        # Try L values: for each divisor d, try L = k*d for k in a range
        # Let's try L in [1, MAX_VAL - W]
        # But that's too many. Let's try L at multiples of the divisors.
        
        found = False
        # Try L = 1, and L = multiples of each divisor
        # For each divisor, try a few multiples
        tried_L = set()
        for d in divisors:
            # Try L = k*d for k such that L is in [1, MAX_VAL - W]
            # Limit the number of tries
            k_start = max(1, (1 + d - 1) // d)
            k_end = (MAX_VAL - W) // d
            # Try a subset: every 10th or so
            step = max(1, (k_end - k_start) // 100) if k_end > k_start else 1
            for k in range(k_start, min(k_end + 1, k_start + 200), step):
                L = k * d
                if L < 1 or L + W > MAX_VAL:
                    continue
                if L in tried_L:
                    continue
                tried_L.add(L)
                
                # Build candidates for each divisor
                candidates = []
                feasible = True
                for i, div in enumerate(divisors):
                    cands = multiples_in_range(div, L, L + W)
                    if not cands:
                        feasible = False
                        break
                    candidates.append(cands)
                
                if not feasible:
                    continue
                
                # Try to find assignment
                result = find_assignment(candidates)
                if result is not None:
                    # Valid solution with width W
                    if best_result is None or W < best_width:
                        best_width = W
                        best_result = result
                        found = True
                        # We can break early if W is small
                        if W <= 30:
                            break
            if found and W <= 30:
                break
        if found and W <= 30:
            break
    
    if best_result is None:
        # Fallback: return the divisors themselves (valid but not optimal)
        return divisors[:]
    
    # Ensure all are Python ints
    return [int(x) for x in best_result]
