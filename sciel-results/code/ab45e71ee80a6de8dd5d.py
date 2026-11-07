"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd
from functools import reduce

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    
    n = len(DIVISORS)
    
    # Compute LCM of all divisors
    lcm_val = 1
    for d in DIVISORS:
        lcm_val = _lcm(lcm_val, d)
    # LCM of [6,10,14,15,21,22,26,33,35,39] = 2*3*5*7*11*13 = 30030
    
    best_range = MAX_VAL
    best_set = None
    
    # Strategy: For each window size W starting from the minimum possible (n-1=9),
    # check if there exists a window [L, L+W] (mod lcm_val) that contains distinct
    # multiples for each divisor. We use bipartite matching to verify feasibility.
    #
    # Key insight from inspiration: iterate W from 9 upward, and for each W,
    # slide L across the period. The number of L values to check can be reduced
    # by only checking "interesting" L values (where window boundaries align with multiples).
    
    # Precompute: for each divisor d, the set of residues (mod lcm_val) that are multiples of d
    # Actually, we just need to check v % d == 0 for candidate values v.
    
    def get_candidates(d, L, W, lcm_val):
        """Get all multiples of d in the circular window [L, L+W] mod lcm_val."""
        # The window covers offsets 0..W from L
        # Value at offset o is (L + o) % lcm_val
        # We need (L + o) % lcm_val to be a multiple of d
        # i.e., (L + o) % lcm_val % d == 0
        # Since lcm_val is a multiple of d, (L + o) % lcm_val % d == (L + o) % d
        # So we need (L + o) % d == 0, i.e., o % d == (-L) % d
        candidates = []
        # Find all o in [0, W] such that (L + o) % d == 0
        # (L + o) % d == 0 iff o % d == (-L) % d
        # Let r = (-L) % d
        # o must satisfy o % d == r and 0 <= o <= W
        # The smallest such o is: if r <= W, then o = r, r+d, r+2d, ...
        # But also o could be r, r+d, ... as long as <= W
        # Wait: o % d == r means o = r + k*d for k >= 0
        # But we also need o >= 0, which is satisfied since r >= 0
        r = (-L) % d
        if r > W:
            # No solution in [0, W]
            # Actually wait: o can also be r - d, r - 2d, ... but those are negative
            # So if r > W, there are no candidates? 
            # Let me re-examine: (L + o) % d == 0
            # L + o = m*d for some integer m
            # o = m*d - L
            # We need 0 <= o <= W
            # So m*d - L >= 0 => m >= ceil(L/d)
            # And m*d - L <= W => m <= floor((L+W)/d)
            # So m ranges from ceil(L/d) to floor((L+W)/d)
            # If ceil(L/d) > floor((L+W)/d), no candidates
            pass
        # Let me use the direct approach:
        # m_min = ceil(L/d), m_max = floor((L+W)/d)
        m_min = (L + d - 1) // d  # ceil(L/d)
        m_max = (L + W) // d       # floor((L+W)/d)
        if m_min > m_max:
            return []
        # Values are m*d for m in [m_min, m_max]
        # But we need them mod lcm_val, and they should be in [0, lcm_val-1]
        # Since m*d might exceed lcm_val, we take mod
        # But actually, since we're working in a circular window of size W < lcm_val,
        # and L is in [0, lcm_val-1], the values (L+o) % lcm_val for o in [0,W]
        # are distinct and cover a contiguous arc.
        # The multiples of d in this arc are exactly m*d % lcm_val for m in [m_min, m_max]
        # But we need to be careful: m*d might be >= lcm_val, so we take mod.
        # However, since W < lcm_val (we'll ensure this), the arc doesn't wrap more than once.
        # Let me just compute the actual values:
        result = []
        for m in range(m_min, m_max + 1):
            v = (m * d) % lcm_val
            # Check that v is actually in the window [L, L+W] (circularly)
            # v is in the window if (v - L) % lcm_val <= W
            if (v - L) % lcm_val <= W:
                result.append(v)
        return result
    
    def has_valid_assignment(L, W, lcm_val, divisors, n):
        """Check if there's a valid assignment of distinct values to divisors using bipartite matching."""
        candidates = []
        for i in range(n):
            cands = get_candidates(divisors[i], L, W, lcm_val)
            candidates.append(cands)
            if not cands:
                return False
        
        # Build bipartite graph
        all_values = set()
        for cands in candidates:
            all_values.update(cands)
        if len(all_values) < n:
            return False
        
        val_list = sorted(all_values)
        val_to_idx = {v: i for i, v in enumerate(val_list)}
        m = len(val_list)
        
        match_to = [-1] * m  # match_to[v_idx] = left node matched to v_idx
        match_from = [-1] * n  # match_from[left] = v_idx matched to left
        
        def try_kun(left, seen):
            for v in candidates[left]:
                v_idx = val_to_idx[v]
                if seen[v_idx]:
                    continue
                seen[v_idx] = True
                if match_to[v_idx] == -1 or try_kun(match_to[v_idx], seen):
                    match_to[v_idx] = left
                    match_from[left] = v_idx
                    return True
            return False
        
        for i in range(n):
            seen = [False] * m
            if not try_kun(i, seen):
                return False
        
        # Reconstruct assignment
        assignment = [0] * n
        for i in range(n):
            v_idx = match_from[i]
            assignment[i] = val_list[v_idx]
        
        return assignment
    
    # Search for minimum W
    for W in range(9, 200):  # Try up to W=199
        if time.time() > deadline:
            break
        
        # For this W, we need to find L such that all divisors have at least one multiple
        # in [L, L+W] and there's a valid matching.
        # 
        # Optimization: we don't need to check all L from 0 to lcm_val-1.
        # The feasibility condition is periodic with period lcm_val, but within one period,
        # we can narrow down candidate L values.
        #
        # For divisor d, the window [L, L+W] contains a multiple of d iff
        # floor((L+W)/d) >= ceil(L/d).
        # This is a periodic condition in L with period d.
        #
        # Let's check L values that are "interesting": L such that L or L+W+1 is a multiple
        # of some divisor, since that's where the feasibility changes.
        
        # Generate candidate L values: for each divisor d, L values where L % d == 0 or (L+W+1) % d == 0
        # i.e., L is a multiple of d, or L+W+1 is a multiple of d
        # L = k*d for k in range, or L = k*d - W - 1 for k in range
        
        candidate_L = set()
        for d in DIVISORS:
            # L such that L % d == 0: L = 0, d, 2d, ... up to lcm_val-1
            for k in range(lcm_val // d + 1):
                L_val = (k * d) % lcm_val
                candidate_L.add(L_val)
            # L such that (L+W+1) % d == 0: L = k*d - W - 1
            for k in range(lcm_val // d + 2):
                L_val = (k * d - W - 1) % lcm_val
                candidate_L.add(L_val)
        
        # Also add L=0 as a baseline
        candidate_L.add(0)
        
        # Shuffle to get a random order? No, let's be systematic.
        # But checking all candidate_L might be too many. Let's limit.
        candidate_L = sorted(candidate_L)
        
        found = False
        for L in candidate_L:
            if time.time() > deadline:
                break
            
            assignment = has_valid_assignment(L, W, lcm_val, DIVISORS, n)
            if assignment is not None:
                # Verify distinctness
                if len(set(assignment)) == n:
                    # Convert to actual positive integers
                    # The values are in [0, lcm_val-1], with 0 representing lcm_val
                    actual = [v if v > 0 else lcm_val for v in assignment]
                    
                    # Verify all are positive and <= MAX_VAL
                    if all(1 <= v <= MAX_VAL for v in actual):
                        # Verify divisibility
                        valid = True
                        for i, d in enumerate(DIVISORS):
                            if actual[i] % d != 0:
                                valid = False
                                break
                        if valid:
                            r = max(actual) - min(actual)
                            if r < best_range:
                                best_range = r
                                best_set = list(actual)
                                found = True
                            if best_range <= 9:
                                break
        
        if found and best_range <= 9:
            break
        
        # If we found a solution with this W, we can stop since we're searching in increasing W order
        # But we might find a better one with the same W at a different L, so continue
        # Actually, once we find ANY solution with span W, we know the minimum is at most W.
        # Since we're iterating W from 9 upward, the first W with a solution is optimal.
        if found:
            # We found a solution with span W, which is the minimum possible
            break
    
    # Fallback: if the above didn't find anything, use the greedy approach from the original
    if best_set is None:
        critical = set()
        for d in DIVISORS:
            for m in range(d, lcm_val + 1, d):
                critical.add(m)
        critical.add(1)
        critical = sorted(critical)
        
        for L in critical:
            if time.time() > deadline:
                break
            
            vals = []
            for i, d in enumerate(DIVISORS):
                q = (L + d - 1) // d
                vals.append(q * d)
            
            seen = set()
            resolved = []
            valid = True
            for i in range(n):
                v = vals[i]
                while v in seen:
                    v += DIVISORS[i]
                    if v > MAX_VAL:
                        valid = False
                        break
                if not valid:
                    break
                seen.add(v)
                resolved.append(v)
            
            if not valid:
                continue
            
            r = max(resolved) - min(resolved)
            if r < best_range:
                best_range = r
                best_set = list(resolved)
                if best_range <= 9:
                    break
    
    if best_set is None:
        best_set = [int(d) for d in DIVISORS]
    
    return [int(v) for v in best_set]
# EVOLVE-BLOCK-END
