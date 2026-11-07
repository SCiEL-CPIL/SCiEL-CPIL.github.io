"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    
    # Calculate LCM of divisors
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    best_span = float('inf')
    best_solution = None
    
    # Deadline guard: 50 seconds to be safe
    start_time = time.time()
    deadline = start_time + 50.0
    
    # We iterate A from 1 to L
    # To speed up, we can step through A, but 30030 is small enough.
    for A in range(1, L + 1):
        if time.time() > deadline:
            break
            
        # Calculate minimum multiples for each divisor >= A
        m = []
        for d in divisors:
            # Smallest multiple of d >= A
            # ceil(A/d) * d
            q = (A + d - 1) // d
            m.append(q * d)
            
        min_m = min(m)
        max_m = max(m)
        span_lb = max_m - min_m
        
        # If the lower bound on span is already >= best found, skip
        if span_lb >= best_span:
            continue
            
        # Try spans from span_lb up to best_span - 1
        # We only need to check if a solution exists with span < best_span
        # We iterate W upwards to find the minimal W for this A
        found = False
        for W in range(span_lb, best_span):
            # Check if there is a system of distinct representatives
            # Candidates for each i: multiples of d_i in [A, A+W]
            candidates = []
            valid = True
            for i, d in enumerate(divisors):
                start_val = m[i]
                # start_val is the smallest multiple >= A
                # We need multiples <= A + W
                cands = []
                v = start_val
                limit = A + W
                while v <= limit:
                    cands.append(v)
                    v += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
                
            if not valid:
                continue
                
            # Sort indices by number of candidates (most constrained first)
            # indices 0..n-1
            idxs = list(range(n))
            idxs.sort(key=lambda i: len(candidates[i]))
            
            # Backtracking
            used = set()
            
            def backtrack(k):
                if k == n:
                    return True
                i = idxs[k]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        if backtrack(k + 1):
                            return True
                        used.remove(val)
                return False
            
            if backtrack(0):
                # Found a solution with span W
                # We need to retrieve the actual values assigned
                # Re-run or store during backtracking?
                # Let's store in a list during backtracking
                # Modification: pass a list to store assignments
                pass # We'll implement retrieval below
                
                # Actually, let's just store the best span and break.
                # We need the values to return.
                # Let's modify backtrack to store values.
                
                # To avoid re-running, we can just find the values again or store them.
                # Since we found it, we can just update best_span and break.
                # But we need the values.
                
                # Let's do a second pass or store in backtrack.
                # For simplicity and speed, we can just return the span and find values later?
                # No, let's store them.
                
                # Let's re-implement backtrack to store values in a list `sol`
                sol = [0] * n
                used.clear()
                
                def backtrack_store(k):
                    if k == n:
                        return True
                    i = idxs[k]
                    for val in candidates[i]:
                        if val not in used:
                            used.add(val)
                            sol[i] = val
                            if backtrack_store(k + 1):
                                return True
                            used.remove(val)
                    return False
                
                if backtrack_store(0):
                    best_span = W
                    best_solution = sol[:]
                    found = True
                    break # Found minimal W for this A, stop checking larger W for this A
                    
        # End W loop
        # If best_span is 1 (minimum possible for distinct ints), we can break?
        # Distinct ints must have span >= 1.
        if best_span == 1:
            break
            
    # If no solution found (should not happen), fallback to divisors
    if best_solution is None:
        return divisors[:]
        
    # Ensure types are int
    return [int(x) for x in best_solution]

# EVOLVE-BLOCK-END
