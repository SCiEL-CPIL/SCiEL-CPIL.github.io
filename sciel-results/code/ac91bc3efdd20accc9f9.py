"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

# Precompute LCM of all divisors
_LCM_ALL = 1
for d in DIVISORS:
    _LCM_ALL = _LCM_ALL * d // gcd(_LCM_ALL, d)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    divisors = DIVISORS
    n = N
    LCM_PERIOD = _LCM_ALL # 30030
    
    # Strategy:
    # 1. Iterate window width W from 1 upwards.
    # 2. For each W, iterate window start L from 1 to LCM_PERIOD.
    # 3. For each window [L, L+W], check if a valid distinct assignment exists.
    # 4. Use backtracking with forward checking for the assignment check.
    
    best_width = 33 # Fallback: span of divisors themselves
    best_assignment = [int(d) for d in divisors]
    
    def solve_window(L, W):
        """
        Check if a valid distinct assignment exists in [L, L+W].
        Returns assignment list if found, else None.
        """
        R = L + W
        candidates = []
        
        # Generate candidate multiples for each divisor in [L, R]
        for d in divisors:
            # First multiple >= L
            k = (L + d - 1) // d
            first = k * d
            if first > R:
                return None
            vals = []
            x = first
            while x <= R:
                vals.append(x)
                x += d
            candidates.append(vals)
        
        # Sort indices by number of candidates (ascending) to prune early
        indices = list(range(n))
        indices.sort(key=lambda i: len(candidates[i]))
        
        used = set()
        assignment = [0] * n
        
        # Backtracking with Forward Checking
        def backtrack(idx):
            if idx == n:
                return True
            
            orig_idx = indices[idx]
            vals = candidates[orig_idx]
            
            for v in vals:
                if v not in used:
                    used.add(v)
                    assignment[orig_idx] = v
                    
                    # Forward Checking: Ensure all remaining unassigned divisors
                    # still have at least one available candidate
                    feasible = True
                    for j in range(idx + 1, n):
                        k_idx = indices[j]
                        has_option = False
                        for cv in candidates[k_idx]:
                            if cv not in used:
                                has_option = True
                                break
                        if not has_option:
                            feasible = False
                            break
                    
                    if feasible and backtrack(idx + 1):
                        return True
                    
                    used.remove(v)
            
            return False
        
        if backtrack(0):
            return assignment
        return None

    # Iterate W from 1 to a reasonable limit.
    # Theoretical minimum is 1, but practically likely 10-20.
    # We stop as soon as we find a valid assignment for a smaller W.
    for W in range(1, 100):
        if time.time() > deadline:
            break
            
        for L in range(1, LCM_PERIOD + 1):
            if time.time() > deadline:
                break
                
            R = L + W
            if R > MAX_VAL:
                break
                
            res = solve_window(L, W)
            if res is not None:
                # Found a valid assignment with width W
                # Since we iterate W from 1, this is the minimum width.
                return [int(x) for x in res]
        
        if time.time() > deadline:
            break
    
    # Fallback: return the divisors themselves (valid, distinct, W=33)
    return best_assignment

# EVOLVE-BLOCK-END
