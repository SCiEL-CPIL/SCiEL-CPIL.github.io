"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = 10
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    divisors = DIVISORS[:]
    n = len(divisors)
    
    # Compute LCM of all divisors
    lcm_val = 1
    for d in divisors:
        lcm_val = _lcm(lcm_val, d)
    
    # Trivial solution
    trivial = divisors[:]
    best_R = max(trivial) - min(trivial)
    best_result = trivial
    
    # Search for smaller R
    # We iterate R from 9 (min possible for 10 distinct ints) upwards.
    # For each R, we check if there exists a window [S, S+R] containing distinct multiples.
    
    for R in range(9, 100):
        if time.time() > deadline - 5.0:
            break
        
        # Precompute valid offsets for each divisor d and residue r
        # valid_offsets[d][r] = list of offsets k in [0, R] such that (r + k) % d == 0
        valid_offsets = {}
        for d in divisors:
            valid_offsets[d] = [[] for _ in range(d)]
            for r in range(d):
                # k such that (r + k) % d == 0 => k % d == (-r) % d
                target = (-r) % d
                # k in [0, R]
                # k = target, target + d, target + 2d, ...
                k = target
                while k <= R:
                    valid_offsets[d][r].append(k)
                    k += d
        
        found_for_R = False
        # Iterate S
        # Optimization: We don't need to check all S if we use the property that 
        # the pattern repeats every LCM. But LCM is 30030, which is small enough.
        for S in range(lcm_val):
            if time.time() > deadline - 5.0:
                break
            
            # Build bipartite graph: Divisors (0..n-1) to Offsets (0..R)
            # adj[i] = list of offsets k such that (S + k) % divisors[i] == 0
            # This is equivalent to k being in valid_offsets[divisors[i]][S % divisors[i]]
            
            adj = [None] * n
            feasible = True
            for i in range(n):
                d = divisors[i]
                r = S % d
                adj[i] = valid_offsets[d][r]
                if not adj[i]:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Kuhn's Algorithm for Max Matching
            # match_offset[o] = divisor index i that matches offset o, or -1
            match_offset = [-1] * (R + 1)
            
            def try_kuhn(v, visited):
                # v is divisor index
                for u in adj[v]:
                    if visited[u]:
                        continue
                    visited[u] = True
                    if match_offset[u] == -1 or try_kuhn(match_offset[u], visited):
                        match_offset[u] = v
                        return True
                return False
            
            # Try to match all divisors
            for i in range(n):
                visited = [False] * (R + 1)
                if not try_kuhn(i, visited):
                    break
            else:
                # All matched
                # Reconstruct assignment
                # match_offset[u] = i means offset u is assigned to divisor i
                # We need assignment[i] = S + u
                assignment = [0] * n
                for u in range(R + 1):
                    if match_offset[u] != -1:
                        i = match_offset[u]
                        assignment[i] = S + u
                
                # Verify distinctness and divisibility (safety check)
                # Distinctness is guaranteed by matching on unique offsets
                # Divisibility is guaranteed by construction
                # Check bounds
                if all(0 < val <= MAX_VAL for val in assignment):
                    actual_R = max(assignment) - min(assignment)
                    if actual_R < best_R:
                        best_R = actual_R
                        best_result = assignment[:]
                    found_for_R = True
                    # Since we iterate R upwards, the first R that works gives the optimal span
                    # (as argued before, if span < R, it would have been found at smaller R)
                    # So we can break
                    break
        
        if found_for_R:
            break
    
    if best_result is None:
        best_result = divisors[:]
    
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
