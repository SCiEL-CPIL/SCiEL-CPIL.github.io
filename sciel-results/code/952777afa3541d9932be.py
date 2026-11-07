import time
import sys
from itertools import permutations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030
    
    # Precompute residues for each divisor
    # residues[i] is a sorted list of r in [0, LCM) such that r % divisors[i] == 0
    residues = []
    for d in divisors:
        res = []
        for r in range(0, LCM, d):
            res.append(r)
        residues.append(res)
        
    # Function to get candidates for divisor i in window [s, s+L]
    # Returns list of actual integers
    def get_candidates(i, s, L):
        d = divisors[i]
        res_list = residues[i]
        # The window is [s, s+L]
        # Multiples are k*d.
        # We need k*d in [s, s+L]
        # So k in [ceil(s/d), floor((s+L)/d)]
        
        # This is faster than filtering the residue list
        k_start = (s + d - 1) // d
        k_end = (s + L) // d
        
        candidates = []
        for k in range(k_start, k_end + 1):
            val = k * d
            # Check if val is within [s, s+L]
            if s <= val <= s + L:
                candidates.append(val)
        return candidates

    # Function to check if SDR exists for a given s and L
    # Uses backtracking
    def check_sdr(s, L):
        # Get candidates for each divisor
        cand_sets = []
        for i in range(n):
            cands = get_candidates(i, s, L)
            if not cands:
                return None
            cand_sets.append(cands)
            
        # Sort divisors by size of candidate set to prune early
        # indices sorted by len(cand_sets[i])
        order = sorted(range(n), key=lambda i: len(cand_sets[i]))
        
        assigned = [None] * n
        used_vals = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            
            i = order[idx]
            for val in cand_sets[i]:
                if val not in used_vals:
                    used_vals.add(val)
                    assigned[i] = val
                    if backtrack(idx + 1):
                        return True
                    used_vals.remove(val)
                    assigned[i] = None
            return False
            
        if backtrack(0):
            return assigned
        return None

    # Binary search on L
    # Lower bound: 1 (since distinct, min range is at least 1, but actually depends on divisors)
    # Upper bound: LCM is safe, but let's start with something reasonable like 100
    lo = 1
    hi = 1000
    
    best_result = None
    best_L = float('inf')
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        
        found = False
        for s in range(LCM):
            if time.time() > deadline:
                break
                
            result = check_sdr(s, mid)
            if result is not None:
                # Verify max and min
                vals = result
                max_val = max(vals)
                min_val = min(vals)
                # The range is max_val - min_val
                # But wait, the window was [s, s+mid].
                # The actual range of the chosen values might be smaller than mid.
                # However, if we found a valid assignment in a window of size mid,
                # the actual range is <= mid.
                # We want to minimize the actual range.
                # So we should record the actual range.
                
                actual_range = max_val - min_val
                if actual_range < best_L:
                    best_L = actual_range
                    best_result = vals[:]
                
                # We found a solution for this L.
                # Can we find a smaller L?
                # Yes, we can set hi = mid - 1 to search for smaller ranges.
                # However, the actual range might be smaller than mid.
                # If we find a solution with actual_range < mid, we should definitely search lower.
                # Even if actual_range == mid, searching lower might yield a tighter fit.
                
                found = True
                break
        
        if found:
            hi = mid - 1
        else:
            lo = mid + 1

    # If no result found within time, return a valid (but maybe not optimal) fallback
    # A simple fallback: just take the divisors themselves (distinct? no, 6,10,14,15,21,22,26,33,35,39 are distinct)
    # But they are not necessarily distinct multiples? Yes, they are distinct integers.
    # Are they <= 1000000? Yes.
    # Do they minimize the range? No, but it's a valid solution.
    if best_result is None:
        # Try a few large L values if binary search didn't finish or failed
        # Or just use the divisors
        best_result = divisors[:]
        best_L = max(divisors) - min(divisors)

    # Final check and conversion
    # Ensure they are ints
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
