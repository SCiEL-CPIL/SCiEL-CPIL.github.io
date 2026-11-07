"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    divisors = DIVISORS
    n = len(divisors)
    L = reduce(_lcm, divisors)
    
    # Bipartite matching helper
    # Left: divisors (0..n-1), Right: candidate values
    def has_matching(candidates):
        """
        candidates: list of lists. candidates[i] is list of values for divisor i.
        Returns True if there is a system of distinct representatives.
        """
        # Map values to indices for the right side
        all_vals = set()
        for c in candidates:
            all_vals.update(c)
        if len(all_vals) < n:
            return False
            
        val_to_idx = {v: i for i, v in enumerate(sorted(all_vals))}
        m = len(all_vals)
        
        # match_to[j] = i (left node matched to right node j)
        match_to = [-1] * m
        # match_from[i] = j (right node matched to left node i)
        match_from = [-1] * n
        
        def try_kun(u, seen):
            for v_val in candidates[u]:
                v_idx = val_to_idx[v_val]
                if seen[v_idx]:
                    continue
                seen[v_idx] = True
                if match_to[v_idx] == -1 or try_kun(match_to[v_idx], seen):
                    match_to[v_idx] = u
                    match_from[u] = v_idx
                    return True
            return False
            
        for i in range(n):
            seen = [False] * m
            if not try_kun(i, seen):
                return False
        return True

    def get_assignment(candidates):
        """
        Returns the list of values if matching exists, else None.
        Re-runs matching to get the actual values.
        """
        all_vals = set()
        for c in candidates:
            all_vals.update(c)
        if len(all_vals) < n:
            return None
            
        val_to_idx = {v: i for i, v in enumerate(sorted(all_vals))}
        idx_to_val = {i: v for v, i in val_to_idx.items()}
        m = len(all_vals)
        
        match_to = [-1] * m
        match_from = [-1] * n
        
        def try_kun(u, seen):
            for v_val in candidates[u]:
                v_idx = val_to_idx[v_val]
                if seen[v_idx]:
                    continue
                seen[v_idx] = True
                if match_to[v_idx] == -1 or try_kun(match_to[v_idx], seen):
                    match_to[v_idx] = u
                    match_from[u] = v_idx
                    return True
            return False
            
        for i in range(n):
            seen = [False] * m
            if not try_kun(i, seen):
                return None
                
        # Extract values
        result = [0] * n
        for i in range(n):
            result[i] = idx_to_val[match_from[i]]
        return result

    # Search for minimal W
    # Minimum span for 10 distinct integers is 9
    max_div = max(divisors)
    
    for W in range(9, 500):  # Cap W at 500, should be found much earlier
        if time.time() > deadline:
            break
            
        # Optimization: X must be such that [X, X+W] contains a multiple of max_div.
        # Multiples of max_div are k * max_div.
        # X <= k * max_div <= X + W  =>  k * max_div - W <= X <= k * max_div
        # We iterate over k such that the interval [k * max_div - W, k * max_div] intersects [0, L-1]
        
        # Generate valid X ranges
        valid_x_ranges = []
        k_min = max(0, (0 - W) // max_div) # k such that k*max_div - W >= 0 is not strictly required, but X>=0
        # Actually, X is in [0, L-1].
        # k * max_div must be in [X, X+W] => X in [k*max_div - W, k*max_div]
        # So we need k such that [k*max_div - W, k*max_div] intersects [0, L-1]
        
        # k * max_div - W <= L - 1 => k <= (L - 1 + W) // max_div
        # k * max_div >= 0 => k >= 0
        
        k_max = (L - 1 + W) // max_div
        for k in range(k_max + 1):
            start = max(0, k * max_div - W)
            end = min(L - 1, k * max_div)
            if start <= end:
                valid_x_ranges.append((start, end))
        
        # Iterate over these ranges
        for start, end in valid_x_ranges:
            for X in range(start, end + 1):
                if time.time() > deadline:
                    return [int(d) for d in divisors] # Safety fallback
                
                candidates = []
                valid = True
                for i in range(n):
                    d = divisors[i]
                    # Find multiples of d in [X, X+W]
                    # First multiple >= X
                    if X % d == 0:
                        first = X
                    else:
                        first = X + (d - X % d)
                    
                    if first > X + W:
                        valid = False
                        break
                        
                    cands = []
                    v = first
                    while v <= X + W:
                        cands.append(v)
                        v += d
                        
                    if not cands:
                        valid = False
                        break
                    candidates.append(cands)
                    
                if not valid:
                    continue
                    
                # Check for SDR
                if has_matching(candidates):
                    # Get the actual assignment
                    assignment = get_assignment(candidates)
                    if assignment:
                        # Ensure types are int
                        return [int(v) for v in assignment]
            
    # Fallback if no solution found within time or W limit
    # This should not happen, but return divisors themselves as valid (though span might be large)
    return [int(d) for d in divisors]
# EVOLVE-BLOCK-END
