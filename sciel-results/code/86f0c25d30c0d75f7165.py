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
    deadline = start_time + 85.0 # 5s margin before 90s limit
    
    # Fallback solution: the divisors themselves
    best_span = max(divisors) - min(divisors)
    best_solution = divisors[:]
    
    # Precompute the "first offset" for each divisor and each residue modulo that divisor.
    # first_offset[i][r] = (d_i - r) % d_i
    first_offsets = []
    for d in divisors:
        offsets = [0] * d
        for r in range(d):
            offsets[r] = (d - r) % d
        first_offsets.append(offsets)
        
    # Iterate W from 1 to best_span - 1.
    # We are looking for the minimum W such that there exists a Y where
    # we can assign distinct multiples.
    
    for W in range(1, best_span):
        if time.time() > deadline:
            break
            
        # Iterate Y from 0 to L-1.
        # Note: Due to periodicity, checking Y in [0, L-1] is sufficient to cover all relative positions.
        
        # Optimization: We can check Y in steps? No, Y determines the exact residues.
        # But we can skip Y if we already checked a "similar" Y?
        # Actually, the pattern of candidates repeats every L.
        
        for Y in range(L):
            if time.time() > deadline:
                break
                
            # Collect candidates for each divisor in [Y, Y+W]
            # candidates[i] is a list of values
            candidates = [None] * n
            valid = True
            
            for i in range(n):
                d = divisors[i]
                rem = Y % d
                offset = first_offsets[i][rem]
                
                if offset > W:
                    valid = False
                    break
                    
                cands = []
                val = Y + offset
                # Number of candidates is floor((W - offset) / d) + 1
                # But we just loop
                while val <= Y + W:
                    cands.append(val)
                    val += d
                    
                candidates[i] = cands
                
            if not valid:
                continue
                
            # Check SDR (System of Distinct Representatives)
            # We need to pick one value from each candidates[i] such that all picked values are distinct.
            
            # Heuristic: Order indices by number of candidates (ascending) to prune early.
            # Indices: 0..n-1
            # Sort order
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used_values = set()
            path = [None] * n
            
            # Backtracking with memoization? 
            # State: (idx, frozenset(used_values))? 
            # used_values can be large, but candidates are small.
            # Actually, since we just need to find ANY valid assignment, and n=10,
            # simple DFS is likely fast enough if the branching factor is low.
            
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
