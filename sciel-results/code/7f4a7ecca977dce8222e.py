import time
import math
from itertools import product

# EVOLVE-BLOCK-START

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of all divisors
    L = divisors[0]
    for i in range(1, n):
        L = _lcm(L, divisors[i])
        
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    # Fallback solution: the divisors themselves
    best_span = max(divisors) - min(divisors)
    best_solution = divisors[:]
    
    # Precompute for each divisor i, the offsets from Y (mod L) where Y+offset is a multiple of d_i
    # offset_i(Y) = (d_i - Y % d_i) % d_i
    # If offset_i(Y) <= W, then Y+offset_i(Y) is a candidate in [Y, Y+W].
    
    # We iterate W from 1 to best_span - 1.
    # For each W, we iterate Y from 0 to L-1.
    # For a given (W, Y), we check if there is a system of distinct representatives (SDR)
    # for the sets S_i = { Y + offset_i(Y) + k*d_i | k >= 0, Y + offset_i(Y) + k*d_i <= Y + W }.
    
    # Optimization:
    # If a window [Y, Y+W] is valid, then any window of size W' > W containing it is also valid.
    # We are looking for the minimum W.
    
    # To speed up the inner loop (checking SDR), we can use a greedy heuristic first,
    # and only fall back to backtracking if needed.
    # Greedy: assign the smallest available candidate to each divisor, prioritizing divisors with fewest candidates.
    
    # Let's precompute the "offset" function for each divisor?
    # offset_i(Y) = (d_i - Y % d_i) % d_i
    # This depends on Y % d_i.
    
    # Since L is 30030, and n is 10, we can afford to compute offsets on the fly if it's fast.
    # But 30030 * 32 * 10 operations is ~10 million, which is fine in Python if optimized?
    # Actually, the backtracking might be the bottleneck.
    
    # Let's try to optimize the SDR check.
    # For small W, the number of candidates per divisor is small (usually 1 or 2).
    # If all have 1 candidate, they must be distinct.
    # If some have 2, we need to check conflicts.
    
    # We will iterate W.
    for W in range(1, best_span):
        if time.time() > deadline:
            break
            
        found = False
        # Iterate Y
        # We can potentially prune Y if we find a pattern?
        # For now, just iterate all Y.
        
        for Y in range(L):
            if time.time() > deadline:
                break
                
            # Collect candidates for each divisor
            # candidates[i] is a list of values in [Y, Y+W] that are multiples of divisors[i]
            candidates = [None] * n
            valid = True
            
            for i in range(n):
                d = divisors[i]
                rem = Y % d
                offset = 0 if rem == 0 else d - rem
                if offset > W:
                    valid = False
                    break
                    
                cands = []
                val = Y + offset
                while val <= Y + W:
                    cands.append(val)
                    val += d
                candidates[i] = cands
                
            if not valid:
                continue
                
            # Check SDR
            # Order indices by number of candidates (ascending)
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used_values = set()
            path = [None] * n
            
            # Backtracking
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for v in candidates[i]:
                    if v not in used_values:
                        used_values.add(v)
                        path[i] = v
                        if backtrack(idx + 1):
                            return True
                        used_values.remove(v)
                        path[i] = None
                return False
            
            if backtrack(0):
                # Found a valid assignment for this W and Y
                # Since we iterate W in increasing order, this is the minimal span.
                # Return the solution
                return [int(x) for x in path]
                
    return [int(x) for x in best_solution]

# EVOLVE-BLOCK-END
