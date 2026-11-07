import math
import time
import numpy as np

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = 10
    max_val = 1000000
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
    
    # Precompute residues for each divisor
    # For divisor d, the residues of multiples of d modulo L are: 0, d, 2d, ..., L-d
    # We need to find 10 distinct residues r_i such that r_i ≡ 0 (mod d_i)
    # and the spread of these residues is minimized.
    
    # Since the numbers must be distinct, and they are multiples of their respective divisors,
    # we can think of this as finding a set of 10 numbers x_i such that:
    # 1. x_i is a multiple of d_i
    # 2. All x_i are distinct
    # 3. The range max(x_i) - min(x_i) is minimized
    
    # Key insight: Since all x_i must be within a window of size W, and they are multiples
    # of their respective divisors, we can search for the minimum window size W.
    
    # For a given window [m, m+W], for each divisor d_i, the valid multiples are:
    # ceil(m/d_i)*d_i, ..., floor((m+W)/d_i)*d_i
    
    # We need to check if we can pick one distinct multiple for each divisor.
    
    # Optimization: Instead of checking all m from 0 to L-1 for each W,
    # we can be smarter about which m values to check.
    
    # For each divisor d_i, the multiples in [m, m+W] depend on m mod d_i.
    # The pattern repeats every L = lcm(all divisors).
    
    # Let's try a different approach: binary search on W, and for each W,
    # check if there exists a valid assignment.
    
    def can_assign(W):
        """Check if there exists a window of size W containing distinct multiples."""
        # For each possible starting residue m (0 to L-1), check if assignment exists
        # But L is large, so we need to be smarter.
        
        # Actually, let's just iterate through possible m values more efficiently.
        # For a window [m, m+W], the number of valid multiples for divisor d is:
        # floor((m+W)/d) - ceil(m/d) + 1 (if positive)
        
        # We need at least one valid multiple for each divisor, and all chosen must be distinct.
        
        # Let's use backtracking with pruning.
        
        # First, check if each divisor has at least one multiple in some window.
        # Actually, we need to check if there's a single window where all can be assigned.
        
        # Iterate through m from 0 to L-1, but with early termination.
        
        for m in range(L):
            if time.time() > deadline:
                return None
            
            # Build candidate lists
            candidates = []
            valid = True
            for d in divisors:
                min_k = (m + d - 1) // d
                max_k = (m + W) // d
                if min_k > max_k:
                    valid = False
                    break
                cands = [k * d for k in range(min_k, max_k + 1)]
                candidates.append(cands)
            
            if not valid:
                continue
            
            # Backtracking to find distinct assignment
            # Sort by number of candidates (MRV heuristic)
            indices = sorted(range(n), key=lambda i: len(candidates[i]))
            chosen = [None] * n
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = indices[idx]
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
                # Found valid assignment
                min_x = min(chosen)
                max_x = max(chosen)
                
                # Check if we can shift to fit in [1, max_val]
                # The assignment is for window [m, m+W]
                # We need to find k such that all x + k*L are in [1, max_val]
                
                # Actually, the assignment found is for specific values in [m, m+W].
                # These values are already absolute positions.
                # We need to check if they're all in [1, max_val].
                
                if min_x >= 1 and max_x <= max_val:
                    return chosen
                
                # If not, we might need to shift by multiples of L
                # But the window is fixed at [m, m+W], so the values are fixed.
                # If they're out of range, this m doesn't work.
                
                # Actually, we can add k*L to all values to shift the window.
                # The new window would be [m+kL, m+W+kL]
                # We need 1 <= m+kL and m+W+kL <= max_val
                # So k >= ceil((1-m)/L) and k <= floor((max_val-m-W)/L)
                
                k_min = math.ceil((1 - min_x) / L)
                k_max = math.floor((max_val - max_x) / L)
                
                if k_min <= k_max:
                    k = k_min
                    return [x + k * L for x in chosen]
        
        return None
    
    # Search for minimum W
    # Lower bound: at least n-1 = 9 (since we need 10 distinct integers)
    # Upper bound: something reasonable
    
    best_result = None
    best_W = float('inf')
    
    # Try small W values first
    for W in range(9, 200):
        if time.time() > deadline:
            break
        
        result = can_assign(W)
        if result is not None:
            # Verify
            if (len(set(result)) == n and 
                all(1 <= x <= max_val for x in result) and 
                all(result[i] % divisors[i] == 0 for i in range(n))):
                best_result = result
                best_W = W
                break  # Found minimum W
    
    if best_result is not None:
        return best_result
    
    # Fallback: return divisors themselves (may not be optimal but valid)
    return divisors[:]
