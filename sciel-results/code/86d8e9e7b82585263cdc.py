"""Optimized search for distinct multiples of divisor set."""
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    best_val = float('inf')
    best_assign = None
    
    # Precompute for each divisor, the residues mod LCM that are "tight"
    # LCM of all divisors
    lcm_all = reduce(_lcm, divisors)
    # lcm_all = 30030
    
    # For each window width W, we need to find if there exists L such that
    # for each i, there is a multiple of divisors[i] in [L, L+W]
    # and all chosen multiples are distinct.
    
    # Strategy: iterate W from small to large. For each W, iterate L mod lcm_all
    # (since the pattern of multiples repeats every lcm_all).
    # For each (W, L), check if a valid assignment exists.
    
    # To speed up: for each divisor d, the multiples in [L, L+W] are:
    # k*d where k*d >= L and k*d <= L+W
    # So k is in [ceil(L/d), floor((L+W)/d)]
    
    # We can use backtracking to assign distinct multiples.
    # Order divisors by number of available multiples in the window (ascending) for pruning.
    
    def can_assign(W, L):
        # For each divisor, compute the list of available multiples in [L, L+W]
        # Sort divisors by number of options (fewest first)
        options = []
        for i, d in enumerate(divisors):
            k_min = (L + d - 1) // d  # ceil(L/d)
            k_max = (L + W) // d       # floor((L+W)/d)
            if k_min > k_max:
                return None  # No multiple for this divisor
            # The multiples are k*d for k in [k_min, k_max]
            # Store (index, list_of_multiples)
            mults = [k * d for k in range(k_min, k_max + 1)]
            options.append((i, mults))
        
        # Sort by number of options
        options.sort(key=lambda x: len(x[1]))
        
        # Backtracking
        used = set()
        result = [None] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(options):
                return True
            i, mults = options[idx]
            for m in mults:
                if m not in used:
                    used.add(m)
                    result[i] = m
                    if backtrack(idx + 1):
                        return True
                    used.remove(m)
                    result[i] = None
            return False
        
        if backtrack(0):
            return result
        return None
    
    # Theoretical minimum W: we need at least 10 distinct values, so W >= 9
    # But also, the largest divisor is 39, so in a window of width W,
    # there's at least one multiple of 39 iff W >= 38 (since gap between multiples is 39).
    # Actually, for a window of width W, a divisor d has a multiple in it
    # iff W >= d-1 (worst case: L just after a multiple, next multiple is d away).
    # So W >= max(divisors) - 1 = 38.
    
    # Let's start W from 9 and go up, but we can skip W < 38 for safety.
    # Actually, let's just start from 9 and let the algorithm handle it.
    
    for W in range(9, 200):
        if time.time() > deadline:
            break
        
        # For each L mod lcm_all, check if assignment exists
        # L can be 0, 1, ..., lcm_all-1 (mod lcm_all)
        # But L must be such that L+W <= MAX_V (or we can wrap around? No, values must be <= MAX_V)
        # Actually, values just need to be <= 1000000. L can be up to MAX_V - W.
        # But due to periodicity, if a solution exists for L, it also exists for L + k*lcm_all.
        # So we only need to check L in [0, lcm_all-1].
        
        for L in range(lcm_all):
            if time.time() > deadline:
                break
            # Check if L + W <= MAX_V (we need the window to fit)
            # Actually, we can always shift the solution up by multiples of lcm_all
            # to fit within MAX_V, as long as L + W + k*lcm_all <= MAX_V for some k.
            # Since lcm_all = 30030 and MAX_V = 1000000, we have plenty of room.
            
            result = can_assign(W, L)
            if result is not None:
                # Found a valid assignment
                # Verify all values are <= MAX_V
                # If not, shift up
                max_val = max(result)
                if max_val > MAX_V:
                    # Shift down? No, L is already small. 
                    # Actually, if L is small, max_val is at most L + W <= lcm_all + W < 1000000
                    # So this shouldn't happen for small W.
                    pass
                
                val = max(result) - min(result)
                if val < best_val:
                    best_val = val
                    best_assign = result[:]
                
                # If we found W, that's the minimum possible for this W
                # But we want the global minimum, so we can break inner loop
                # and continue to next W? No, we already have W, so val = W.
                # Actually, val might be less than W if the min and max are not at the edges.
                # But since we're iterating W from small to large, the first valid W gives the best.
                # Wait, no: for a given W, the actual range might be less than W.
                # But we're minimizing max-min, which is <= W.
                # Since we're trying W in increasing order, the first W that works
                # gives a solution with range <= W. But maybe a larger W gives a smaller range?
                # No, because if a solution has range R, it fits in a window of width R.
                # So the minimum range is the minimum W for which a solution exists.
                # Therefore, we can break both loops once we find the first valid W.
                
                # But wait: is it possible that for W=10, the best range is 10,
                # but for W=11, the best range is 9? No, because if there's a solution
                # with range 9, it fits in a window of width 9, so W=9 would have worked.
                
                # So the first valid W gives the optimal range.
                break
        
        if best_assign is not None:
            # We found a solution for this W, which is the minimum possible
            break
    
    if best_assign is None:
        # Fallback: use the divisors themselves
        best_assign = divisors[:]
    
    # Ensure types are correct
    return [int(x) for x in best_assign]
