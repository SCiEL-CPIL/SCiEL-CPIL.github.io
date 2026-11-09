"""Rewritten program: SEARCH approach to pack distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def _lcm(a, b):
    return a * b // gcd(a, b)

def _find_multiple_in_window(d, lo, hi):
    """Find a multiple of d in [lo, hi]. Returns list of possible multiples (capped)."""
    if hi < lo:
        return []
    # First multiple >= lo
    first = ((lo + d - 1) // d) * d
    if first > hi:
        return []
    # Collect multiples (cap to avoid huge lists)
    mults = []
    m = first
    while m <= hi and len(mults) < 200:
        mults.append(m)
        m += d
    return mults

def _solve_window(divisors, lo, hi):
    """Try to assign distinct multiples to each divisor within [lo, hi].
    Returns assignment list or None."""
    n = len(divisors)
    # For each divisor, get candidate multiples in window
    candidates = []
    for d in divisors:
        cands = _find_multiple_in_window(d, lo, hi)
        if not cands:
            return None
        candidates.append(cands)
    
    # Backtracking search with ordering by fewest candidates first
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    assignment = [None] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in assignment:
                assignment[i] = val
                if backtrack(idx + 1):
                    return True
                assignment[i] = None
        return False
    
    if backtrack(0):
        return assignment
    return None

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    best_val = float('inf')
    best_assign = None
    
    # Strategy: search over window widths W, starting small
    # For each W, try random starting positions L
    # Also try systematic approaches
    
    # Precompute: for a window of width W, each divisor d has ~W/d multiples
    # We need at least 1 for each, and all distinct
    
    # Start with small windows and increase
    # The minimum possible window is at least n-1 (since values are distinct integers)
    
    # Try window widths from small to large
    # For efficiency, use random sampling of L positions
    
    # First, try a greedy/heuristic approach for small windows
    # Then expand
    
    # Sort divisors by size to help with backtracking
    # But we need to return in original order
    
    # Let's try: for W in range(n-1, 500):
    #   try random L values
    
    random.seed(42)
    
    # Try increasing window widths
    for W in range(n - 1, 500):
        if time.time() > deadline - 5:
            break
        
        # Number of trials for this width
        # More trials for smaller widths
        trials = max(10, 200 - W)
        
        for _ in range(trials):
            if time.time() > deadline - 5:
                break
            
            # Random start position
            lo = random.randint(1, MAX_V - W)
            hi = lo + W
            
            assign = _solve_window(divisors, lo, hi)
            if assign is not None:
                # Verify distinctness
                if len(set(assign)) == n:
                    val = max(assign) - min(assign)
                    if val < best_val:
                        best_val = val
                        best_assign = assign[:]
                        if best_val == n - 1:  # theoretical minimum for distinct ints
                            break
                    # If we found a good one, we can break early for this W
                    if val <= W + 2:
                        break
    
    # If we found a solution, return it
    if best_assign is not None:
        return list(best_assign)
    
    # Fallback: return the divisors themselves (valid but not optimal)
    return list(divisors)
# EVOLVE-BLOCK-END
