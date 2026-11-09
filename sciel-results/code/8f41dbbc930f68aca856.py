"""Rewritten program: DIRECT approach to pack distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Calculate LCM of all divisors to exploit periodicity
    def lcm(a, b):
        return a * b // gcd(a, b)
    lcm_all = reduce(lcm, divisors)
    
    # We search for the minimum span W.
    # Since the pattern of divisibility repeats every lcm_all, 
    # we only need to check window starts L in [1, lcm_all].
    
    # Optimized CSP solver with MRV and Forward Checking
    def find_assignment(candidates):
        n = len(candidates)
        # Check for immediate failure
        for c in candidates:
            if len(c) == 0:
                return None
        
        # Deep copy candidates for forward checking
        cands = [list(c) for c in candidates]
        assignment = [None] * n
        used = set()
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Find unassigned variable with minimum remaining values (MRV)
            min_var = -1
            min_count = float('inf')
            for i in range(n):
                if assignment[i] is None:
                    if len(cands[i]) < min_count:
                        min_count = len(cands[i])
                        min_var = i
                        if min_count == 0:
                            return False
            
            if min_var == -1:
                return True # All assigned
            
            i = min_var
            
            # Try each candidate value
            # Iterate over a copy or handle removal carefully? 
            # We are modifying cands[j] for j != i, not cands[i] directly in this loop
            # But we need to be careful if we remove from cands[i]? No, we don't remove from cands[i] in forward check.
            for val in cands[i]:
                if val in used:
                    continue
                
                # Assign
                assignment[i] = val
                used.add(val)
                
                # Forward Checking
                saved_changes = []
                valid = True
                for j in range(n):
                    if j == i or assignment[j] is not None:
                        continue
                    if val in cands[j]:
                        saved_changes.append(j)
                        cands[j].remove(val)
                        if len(cands[j]) == 0:
                            valid = False
                            break
                
                if valid:
                    if backtrack():
                        return True
                
                # Undo
                assignment[i] = None
                used.remove(val)
                for j in saved_changes:
                    cands[j].append(val)
                    # Note: Order in cands[j] is not restored, but membership is what matters.
                    # However, if we rely on order for anything, this is a bug.
                    # The problem does not depend on candidate order.
                    
            return False
            
        if backtrack():
            return assignment
        return None

    # Iterate span W from minimum possible (9) upwards
    # The maximum span is bounded by the divisors, but 1000 is a safe upper bound for this small set
    for W in range(9, 1000):
        if time.time() > deadline:
            break
        
        # Iterate window start L
        # We only need to check L up to lcm_all because the pattern repeats
        for L in range(1, lcm_all + 1):
            if time.time() > deadline:
                break
            
            # Check if window [L, L+W] is within bounds
            # Actually, we can wrap or just check. 
            # The problem says "positive integers <= 1000000".
            # If L+W > MAX_V, we might need to handle wrap-around or skip.
            # Given lcm_all is ~30k and W is small, L+W will be well under MAX_V 
            # for the first few valid L.
            if L + W > MAX_V:
                continue

            candidates = []
            valid_window = True
            
            for d in divisors:
                # Find multiples of d in [L, L+W]
                first_k = (L + d - 1) // d
                last_k = (L + W) // d
                
                if first_k > last_k:
                    valid_window = False
                    break
                
                # Generate the list of candidates
                # Optimization: use list comprehension
                cands = [k * d for k in range(first_k, last_k + 1)]
                candidates.append(cands)
            
            if not valid_window:
                continue
            
            # Try to assign distinct values
            result = find_assignment(candidates)
            
            if result is not None:
                # Verify the solution (safety check)
                if len(set(result)) != n:
                    continue
                for i, d in enumerate(divisors):
                    if result[i] % d != 0:
                        continue
                if any(v < 1 or v > MAX_V for v in result):
                    continue
                
                # Return the result
                return [int(v) for v in result]
        
        # If we finish all L for a given W without success, increment W
    
    # Fallback: Return a valid solution if not found (should not happen)
    # Using the divisors themselves (might not be distinct if duplicates existed, but they are distinct here)
    # Actually, divisors are distinct.
    fallback = [int(d) for d in divisors]
    return fallback

# EVOLVE-BLOCK-END
