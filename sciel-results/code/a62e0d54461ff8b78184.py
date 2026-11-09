"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_all(nums):
    return reduce(_lcm, nums, 1)

def construct():
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    period = _lcm_all(divisors)  # 30030
    
    deadline = time.time() + 80.0  # Safety margin before 90s limit
    
    best_solution = None
    best_span = float('inf')
    
    # For each window width W, search for a valid L
    # The window [L, L+W] must contain at least one multiple of each divisor
    # and we need a perfect matching (distinct values)
    
    # Precompute: for each divisor, the possible offsets within a period
    # A value v is a multiple of d iff v % d == 0
    # In window [L, L+W], multiples of d are at positions:
    #   first = L + ((d - L % d) % d)
    #   then every d steps
    
    def check_window(L, W):
        """Check if window [L, L+W] has a valid assignment.
        Returns the assignment (list of values) or None."""
        if L < 1 or L + W > max_val:
            return None
        
        # For each divisor i, compute the list of multiples in [L, L+W]
        # We'll use backtracking with constraint propagation
        
        # Build candidate lists for each divisor
        candidates = []
        for i, d in enumerate(divisors):
            # First multiple of d >= L
            rem = L % d
            if rem == 0:
                first = L
            else:
                first = L + (d - rem)
            cands = []
            v = first
            while v <= L + W:
                cands.append(v)
                v += d
            candidates.append(cands)
            if len(cands) == 0:
                return None
        
        # Backtracking with MRV (minimum remaining values) heuristic
        # Order divisors by number of candidates (ascending)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        assignment = [None] * n
        used_values = set()
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            i = order[idx]
            for v in candidates[i]:
                if v not in used_values:
                    used_values.add(v)
                    assignment[i] = v
                    if backtrack(idx + 1):
                        return True
                    used_values.discard(v)
                    assignment[i] = None
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    # Search strategy:
    # 1. Try small W values first
    # 2. For each W, try different L values (within one period)
    # 3. The window must be within [1, max_val]
    
    # Minimum possible W: at least n-1 = 9 (since values must be distinct)
    # But also constrained by the divisors' spacing
    
    # Start with W = 9 and increase
    # For each W, try L values that are "aligned" with the divisors
    # Key insight: L mod period determines the pattern, so we only need to check
    # L in [1, period] for each W, then extend to valid range
    
    # Actually, for a given W, we need L such that [L, L+W] ⊆ [1, max_val]
    # So L ∈ [1, max_val - W]
    # Due to periodicity, the pattern repeats every period
    # So we can check L in [1, min(period, max_val - W)] and if found, we're done
    
    # But we also need to be careful: the window [L, L+W] might wrap around
    # in a way that's only valid for certain L mod period
    
    # Let's be more systematic:
    # For W from 9 upward:
    #   For L from 1 to min(period, max_val - W):
    #     Check if [L, L+W] is valid
    #   If found, return
    
    # Optimization: use bitsets for faster checking
    # But let's start with the backtracking approach and see if it's fast enough
    
    # Pre-filter: for a window to be valid, each divisor must have at least one
    # multiple in the window. The minimum W for divisor d is 0 (if L is a multiple)
    # but typically we need W >= d-1 to guarantee a multiple
    
    # Let's try W from a reasonable lower bound
    # Lower bound: max(d_i - 1) for the tightest constraint? No, that's not right.
    # Actually, for each divisor d, the window must contain at least one multiple of d.
    # The worst case is when L is just after a multiple, so we need W >= d-1.
    # So W_min = max(d_i - 1) = 39 - 1 = 38? No, that's too pessimistic.
    # We just need SOME multiple in the window, not that every position works.
    
    # Let's start with W = 9 and go up
    found = False
    for W in range(9, 200):  # Try up to W=200
        if time.time() > deadline:
            break
        
        # For this W, try L values
        # Due to periodicity, we can limit L to [1, period]
        # But we also need L + W <= max_val
        max_L = min(period, max_val - W)
        
        if max_L < 1:
            continue
        
        # Try L values; we can be smarter by focusing on L values that
        # align well with the divisors
        # For now, try all L in [1, max_L] but step by 1
        # This could be slow for large period... period = 30030
        
        # Optimization: only try L where at least one divisor has a multiple
        # at L or L+1, etc. But let's just try all and see.
        
        # Actually, let's be smarter: for each L, the set of available multiples
        # depends on L mod d for each d. So we can group L values by their
        # residue vector (L mod d for each d). There are at most period distinct
        # residue vectors.
        
        # For speed, let's try a subset of L values first
        # Focus on L values that are multiples of small divisors or close to them
        
        # Try L = 1, 2, ..., max_L but with early termination
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
            
            sol = check_window(L, W)
            if sol is not None:
                span = max(sol) - min(sol)
                if span < best_span:
                    best_span = span
                    best_solution = sol[:]
                found = True
                break  # Found a solution for this W, move to next W
        
        if found and best_span == W:
            # Can't do better than W for this W, and we're searching in order
            break
    
    if best_solution is not None:
        # Ensure types are correct
        return [int(v) for v in best_solution]
    
    # Fallback: return the divisors themselves (valid but not optimal)
    return P['divisors'][:]

# EVOLVE-BLOCK-END
