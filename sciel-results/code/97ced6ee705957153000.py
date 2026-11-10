import time
import sys

# EVOLVE-BLOCK-START
import math

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def construct():
    start_time = time.time()
    deadline = 55.0  # seconds, keep margin
    
    # We search for the minimum window width W
    # Start from W=1 and increase
    
    # Pre-compute nothing, just iterate
    
    # To ensure we return a valid result even if we time out,
    # we can keep track of the first valid solution found for any W.
    # But since we increase W, the first one is optimal.
    # If we time out, we might not find it.
    # However, the search is very fast for small W.
    
    # Optimization: 
    # For a given W, L is valid only if for all d, there is a multiple in [L, L+W].
    # This is equivalent to: (-L) % d <= W for all d.
    
    for W in range(1, 200):
        if time.time() - start_time > deadline:
            break
            
        # Iterate L from 1 to MAX_VAL - W
        # We can stop early if we find a solution.
        
        # Optimization: 
        # We can iterate L in steps? No, L is integer.
        # But we can skip L if it fails the modulo check.
        
        # Pre-compute valid residues for this W?
        # valid_residues[d] = set of r in [0, d-1] such that (-r) % d <= W
        # Actually, (-L) % d is the distance to the next multiple.
        # Let dist = (-L) % d. We need dist <= W.
        # So L % d must be in {d - dist | dist in [0, min(W, d-1)]} U {0} (if dist=0)
        # Specifically, if dist=0, L%d=0.
        # If dist>0, L%d = d - dist.
        # So L%d in {0, d-W, d-W+1, ..., d-1} (if W < d)
        # Or L%d in {0, 1, ..., W} (if W >= d, all residues are valid)
        
        valid_residues = []
        for d in DIVISORS:
            if W >= d - 1:
                # All residues are valid
                valid_residues.append(None) 
            else:
                # Residues r such that (-r) % d <= W
                # dist = 0 => r=0
                # dist = 1 => r=d-1
                # ...
                # dist = W => r=d-W
                # So r in {0, d-W, d-W+1, ..., d-1}
                # This is a contiguous range [d-W, d-1] plus {0}
                # Note: if W=0, only 0 is valid.
                # If W=1, {0, d-1}
                valid_residues.append((d - W, d - 1)) # range inclusive, plus 0
        
        # Iterate L
        # We can iterate L from 1 to MAX_VAL - W
        # To speed up, we can jump?
        # The condition is L % d_i in S_i.
        # This is a system of congruences-like constraint, but "in set".
        # We can just iterate.
        
        # To avoid Python loop overhead for 10^6 iterations * 100 W's = 10^8, 
        # we need to be careful.
        # 10^8 simple ops in Python is ~10-20s. 
        # But we break as soon as we find a solution.
        # Expected W is small.
        
        # Let's add a time check inside the L loop.
        
        for L in range(1, MAX_VAL - W + 1):
            if time.time() - start_time > deadline:
                # Fallback: return a valid solution if we have one, else try a heuristic
                # Since we are in the middle of a W, we might not have a solution for this W.
                # But we might have found one in a previous W? No, we return immediately.
                # So if we time out, we have no solution.
                # We must return a valid solution.
                # We can use a greedy approach to find a valid solution with large W.
                # Or just return the divisors themselves (which are distinct and valid, but W is large).
                # Divisors: 6, 10, 14, 15, 21, 22, 26, 33, 35, 39.
                # Max 39, Min 6. W=33.
                # This is a valid fallback.
                return DIVISORS[:]
                
            # Check if L is a candidate start
            valid = True
            for i, d in enumerate(DIVISORS):
                r = L % d
                vr = valid_residues[i]
                if vr is None:
                    continue
                # Check if r is 0 or in [vr[0], vr[1]]
                if r != 0 and not (vr[0] <= r <= vr[1]):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # If valid, we need to assign distinct values.
            # Gather candidates
            avail = []
            for d in DIVISORS:
                # First multiple >= L
                # dist = (-L) % d
                # first_m = L + dist
                first_m = L + ((-L) % d)
                ms = []
                m = first_m
                while m <= L + W:
                    ms.append(m)
                    m += d
                avail.append(ms)
                
            # Backtracking
            # Sort by number of options
            indices = list(range(len(DIVISORS)))
            indices.sort(key=lambda i: len(avail[i]))
            
            used = set()
            assignment = [None] * len(DIVISORS)
            
            # Recursive backtracking
            # To avoid recursion limit issues (unlikely for 10 items), 
            # and to speed up, we can implement iteratively or just use recursion.
            # Recursion depth 10 is fine.
            
            def bt(idx):
                if idx == len(indices):
                    return True
                i = indices[idx]
                for m in avail[i]:
                    if m not in used:
                        used.add(m)
                        assignment[i] = m
                        if bt(idx + 1):
                            return True
                        used.remove(m)
                        assignment[i] = None
                return False
            
            if bt(0):
                # Convert to list of ints
                return [int(x) for x in assignment]
        
    # If loop finishes without returning, return fallback
    return DIVISORS[:]
# EVOLVE-BLOCK-END
