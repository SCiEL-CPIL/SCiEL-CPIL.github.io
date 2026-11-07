"""
Strategy:
1. Determine the LCM of the divisors to understand the periodicity of the problem.
2. We want to find 10 distinct integers x_i such that x_i % d_i == 0.
3. Let the interval be [L, L+W]. We want to minimize W.
4. The key insight is that for large divisors (d > W), there is at most one multiple in any window of size W.
   These "rigid" divisors have fixed values for a given L (if they exist in the window).
   The "flexible" divisors (d <= W) can be assigned via backtracking/matching.

5. The pattern of divisibility repeats every LCM = 30030.
   So we only need to check L in [0, 30029].

6. Algorithm:
   - Iterate W from 0 upwards.
   - For each W, iterate L from 0 to LCM-1.
   - For each (L, W), check if a valid assignment exists.
   - As soon as we find a valid (L, W), that W is the minimum possible (since we iterate W in increasing order).
   - Return the corresponding solution.

7. Optimization:
   - For rigid divisors (d > W), the valid L residues mod d are precomputed.
   - We can quickly reject L values where any rigid divisor has no multiple in [L, L+W].
   - For the remaining L values, we use backtracking for the flexible divisors.
"""

import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin (evaluator kills at 90s)
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)  # 30030
    
    best_W = 33  # Seed span
    best_solution = None
    
    # Precompute offset tables for each divisor
    # offset_table[d][rem] = offset (val - L) if valid, else -1
    # For a given L with L % d = rem, the first multiple >= L is:
    #   val = L + (d - rem if rem != 0 else 0)
    # This is in [L, L+W] iff offset <= W
    
    # We'll compute these on the fly for each W since the condition depends on W
    
    for W in range(0, best_W + 1):
        if time.time() > deadline:
            break
            
        # Precompute rigid and flexible indices for this W
        rigid_indices = []
        flexible_indices = []
        for i, d in enumerate(divisors):
            if d > W:
                rigid_indices.append(i)
            else:
                flexible_indices.append(i)
        
        # Precompute offset tables for rigid divisors
        # For each rigid divisor d, table[rem] = offset if valid else -1
        offset_tables = []
        for i in rigid_indices:
            d = divisors[i]
            table = [-1] * d
            for rem in range(d):
                offset = d - rem if rem != 0 else 0
                if offset <= W:
                    table[rem] = offset
            offset_tables.append(table)
        
        # Iterate L from 0 to LCM-1
        for L in range(0, LCM):
            if time.time() > deadline:
                break
                
            # Check rigid divisors
            used_offsets = set()
            assignments = [None] * n
            valid = True
            
            for ri, i in enumerate(rigid_indices):
                d = divisors[i]
                rem = L % d
                offset = offset_tables[ri][rem]
                if offset < 0:
                    valid = False
                    break
                if offset in used_offsets:
                    valid = False
                    break
                used_offsets.add(offset)
                assignments[i] = L + offset
            
            if not valid:
                continue
            
            # Check flexible divisors via backtracking
            sol = _assign_flexible(divisors, flexible_indices, L, W, used_offsets, assignments)
            if sol is not None:
                best_W = W
                best_solution = sol
                break  # Found min W for this W, so this is the global min W
        
        if best_solution is not None:
            break
            
    if best_solution is None:
        # Fallback to seed
        return divisors[:]
        
    # Ensure all values are Python int and within bounds
    result = []
    for i, val in enumerate(best_solution):
        if val is None:
            return divisors[:]
        v = int(val)
        if not (1 <= v <= 1000000) or v % divisors[i] != 0:
            return divisors[:]
        result.append(v)
    
    # Check distinctness
    if len(set(result)) != len(result):
        return divisors[:]
    
    return result

def _assign_flexible(divisors, flexible_indices, L, W, used_offsets, assignments):
    """
    Assign flexible divisors via backtracking.
    used_offsets contains offsets (val - L) already used by rigid divisors.
    Returns the full assignments list if successful, None otherwise.
    """
    if not flexible_indices:
        return assignments
    
    # Generate candidates for each flexible index
    # Candidate is an offset: val = L + offset, where val % d == 0
    candidates = []
    for i in flexible_indices:
        d = divisors[i]
        cands = []
        k = (L + d - 1) // d
        val = k * d
        while val <= L + W:
            offset = val - L
            if offset not in used_offsets:
                cands.append(offset)
            val += d
        candidates.append(cands)
    
    # If any flexible divisor has no candidates, fail
    for cands in candidates:
        if not cands:
            return None
    
    # Order by fewest candidates (MRV)
    order = sorted(range(len(flexible_indices)), key=lambda x: len(candidates[x]))
    
    def backtrack(idx):
        if idx == len(order):
            return True
        flex_idx_in_list = order[idx]
        original_idx = flexible_indices[flex_idx_in_list]
        for offset in candidates[flex_idx_in_list]:
            if offset not in used_offsets:
                used_offsets.add(offset)
                assignments[original_idx] = L + offset
                if backtrack(idx + 1):
                    return True
                used_offsets.remove(offset)
                assignments[original_idx] = None
        return False
        
    if backtrack(0):
        return assignments
    else:
        return None

# EVOLVE-BLOCK-END
