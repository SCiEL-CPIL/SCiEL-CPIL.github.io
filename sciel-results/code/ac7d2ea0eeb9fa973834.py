"""Rewritten program for packing distinct multiples."""
import time
import sys

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # We search for the minimum range L
    # Start with a small L and increase
    # Based on the divisors, a range of ~20-30 is likely sufficient
    for L in range(10, 200):
        # Check time
        if time.time() > deadline:
            break
            
        # We scan potential minimums 'm'
        # The pattern of valid windows repeats every LCM of divisors, but that's huge.
        # However, for a fixed L, if a solution exists, it likely exists for small m 
        # or we can just scan a reasonable range of m.
        # Let's scan m from 1 to 5000.
        for m in range(1, 5001):
            # Check time
            if time.time() > deadline:
                break
                
            # Determine candidates for each divisor
            candidates = []
            feasible = True
            for d in DIVISORS:
                # Find multiples of d in [m, m+L]
                # First multiple >= m: k_min = ceil(m/d)
                # Last multiple <= m+L: k_max = floor((m+L)/d)
                k_min = (m + d - 1) // d
                k_max = (m + L) // d
                
                if k_min > k_max:
                    feasible = False
                    break
                
                # Generate the actual values
                # To optimize backtracking, sort candidates by size
                # But we need to keep track of which divisor they belong to
                vals = [k * d for k in range(k_min, k_max + 1)]
                candidates.append((len(vals), vals))
            
            if not feasible:
                continue
                
            # Sort by number of candidates (most constrained first)
            # This helps backtracking prune early
            # We need to map back to original indices
            sorted_candidates = sorted(candidates, key=lambda x: x[0])
            
            # Backtracking search
            # We assign values to positions in sorted_candidates order
            # But we need to return the result in the original order
            # Let's use a simple recursive backtracker
            used = set()
            assignment = [None] * len(DIVISORS)
            
            # Map sorted index to original index
            # sorted_candidates[i] corresponds to some original divisor
            # We need to know which original index it is.
            # Let's restructure:
            # Create list of (size, vals, orig_idx)
            candidates_info = []
            for i, d in enumerate(DIVISORS):
                k_min = (m + d - 1) // d
                k_max = (m + L) // d
                if k_min > k_max:
                    candidates_info = None
                    break
                vals = [k * d for k in range(k_min, k_max + 1)]
                candidates_info.append((len(vals), vals, i))
            
            if candidates_info is None:
                continue
                
            candidates_info.sort(key=lambda x: x[0])
            
            def backtrack(idx, used_vals):
                if idx == len(candidates_info):
                    return True
                size, vals, orig_idx = candidates_info[idx]
                for v in vals:
                    if v not in used_vals:
                        used_vals.add(v)
                        assignment[orig_idx] = v
                        if backtrack(idx + 1, used_vals):
                            return True
                        used_vals.remove(v)
                        assignment[orig_idx] = None
                return False
            
            if backtrack(0, set()):
                # Found a solution
                # Verify types and constraints
                result = []
                for i in range(len(DIVISORS)):
                    x = assignment[i]
                    if x is None or x % DIVISORS[i] != 0 or x < 1 or x > MAX_VAL:
                        return DIVISORS[:] # Fallback
                    result.append(int(x))
                
                # Check distinctness
                if len(set(result)) != len(result):
                    return DIVISORS[:]
                    
                return result
            
    # Fallback if no solution found within time/bounds
    # This should not happen for L up to 200
    return DIVISORS[:]
# EVOLVE-BLOCK-END
