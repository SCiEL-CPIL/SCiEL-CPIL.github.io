import time
import math

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
    
    # Fallback solution
    best_span = 33
    best_solution = divisors[:]
    
    # Precompute min_offset[Y][i] for all Y in [0, L) and i in [0, n)
    # min_offset[Y][i] is the smallest non-negative k such that (Y+k) % divisors[i] == 0
    # k = (divisors[i] - Y % divisors[i]) % divisors[i]
    
    # To save memory and time, we can compute this on the fly or store it.
    # L=30030, n=10. 300300 integers. Small.
    # But we can just compute it inside the loop if we are fast.
    # Actually, let's precompute a list of lists `min_offsets` of size L x n.
    # Or better, for each Y, compute the max min_offset.
    
    # We will iterate W from 1 to best_span - 1.
    # For each W, we need to find Y such that for all i, min_offset[Y][i] <= W.
    # And then check if distinct assignment is possible.
    
    # Optimization:
    # Sort divisors by value descending to put constraints with fewer candidates first in backtracking.
    # But the indices in the solution must match the original divisor order.
    # So we keep a mapping.
    
    # Let's precompute the candidate offsets for each divisor for a given W?
    # No, W changes.
    
    # Let's iterate W.
    for W in range(1, best_span):
        if time.time() > deadline:
            break
            
        # For each Y, check if window is valid (all divisors have a multiple)
        # If valid, check distinctness.
        
        # To speed up, we can precompute for each Y the list of candidates for each divisor
        # for the current W.
        
        # However, generating candidates for all Y and all divisors for each W might be costly if W is large.
        # But W is small (<= 32).
        
        # Let's try to just loop Y and check.
        for Y in range(L):
            if time.time() > deadline:
                break
                
            # Check if window [Y, Y+W] has a multiple for each divisor
            # and collect candidates
            candidates = [None] * n
            valid = True
            
            for i in range(n):
                d = divisors[i]
                # First multiple >= Y
                # k = ceil(Y/d)
                # val = k * d
                # If val > Y + W, no candidate.
                
                # Offset of first multiple from Y
                rem = Y % d
                offset = 0 if rem == 0 else d - rem
                if offset > W:
                    valid = False
                    break
                    
                # Generate candidates
                cands = []
                val = Y + offset
                while val <= Y + W:
                    cands.append(val)
                    val += d
                candidates[i] = cands
                
            if not valid:
                continue
                
            # Check distinctness using backtracking
            # Order indices by number of candidates (ascending) to prune early
            # But we need to return result in original order.
            
            # Create list of (len(candidates[i]), i)
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used_values = set()
            path = [None] * n
            
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
                # Found a valid assignment
                # The span is max(path) - min(path)
                # It is <= W.
                # Since we iterate W in increasing order, this is the minimal span?
                # Yes, because any solution with span S < W would have been found in earlier iterations.
                # Wait, the window size is W. The actual span of the chosen numbers is max-min.
                # If we find a solution in window W, the span is <= W.
                # Could it be < W? Yes.
                # But if span < W, it fits in a smaller window.
                # So it would have been found when W' = span.
                # So the first W we find a solution for gives the minimal span.
                
                return [int(x) for x in path]
                
    return [int(x) for x in best_solution]

# EVOLVE-BLOCK-END
