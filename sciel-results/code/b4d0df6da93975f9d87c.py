"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Returns the set of denominators for the Egyptian fraction expansion of 5/31
    that minimizes the largest denominator.
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    target = Fraction(5, 31)
    best_max_den = 248  # current known solution
    best_solution = [8, 31, 248]
    
    # Try to find solutions with smaller max denominator
    # Search over number of terms from 3 up, and max denominator from small to large
    # We'll do a bounded search
    
    def search(num_terms, max_den, min_den=1, used=None):
        nonlocal best_max_den, best_solution
        if used is None:
            used = set()
        
        # Pruning: if we've already exceeded our best, stop
        if max_den >= best_max_den:
            return
        
        # Check time
        if time.time() > deadline:
            return
        
        # Try all possible values for the first denominator
        # The remaining (num_terms-1) terms must sum to target - 1/d
        # Each of those terms is at most 1/(min_den+1) ... but we need to be smarter
        
        for d in range(min_den, max_den + 1):
            if d in used:
                continue
            if time.time() > deadline:
                return
            
            remaining = target - Fraction(1, d)
            if remaining <= 0:
                continue
            
            if num_terms == 1:
                if remaining == 0:
                    # Found a complete solution
                    sol = sorted(list(used | {d}))
                    if max(sol) < best_max_den:
                        best_max_den = max(sol)
                        best_solution = sol
                continue
            
            # For the remaining terms, the smallest possible sum is (num_terms-1) * 1/max_den
            # The largest possible sum is (num_terms-1) * 1/(d+1) approximately
            # More precisely, remaining terms must each be >= d+1 (since distinct and > d)
            # Max sum of remaining: (num_terms-1) * 1/(d+1)
            # Min sum of remaining: (num_terms-1) * 1/max_den
            if remaining > Fraction(num_terms - 1, d + 1):
                continue
            if remaining < Fraction(num_terms - 1, max_den):
                # Even the smallest possible remaining terms sum to more than needed
                # Actually wait, we want remaining to be achievable
                # The remaining terms are >= d+1, so each <= 1/(d+1)
                # So max possible sum of remaining = (num_terms-1)/(d+1)
                # Min possible sum of remaining = (num_terms-1)/max_den
                # We need remaining to be in [min_possible, max_possible]
                if remaining < Fraction(num_terms - 1, max_den):
                    # remaining is too small - even using the largest denominators we can't get that small
                    # Actually no: smaller denominators give larger reciprocals
                    # So min sum uses largest denominators: (num_terms-1)/max_den
                    # If remaining < (num_terms-1)/max_den, we can't achieve it
                    continue
                # If remaining > (num_terms-1)/(d+1), we can't achieve it (checked above)
            
            # Recurse
            new_used = used | {d}
            search(num_terms - 1, max_den, d + 1, new_used)
    
    # Try 3 terms with decreasing max_den
    for max_d in range(247, 50, -1):
        if time.time() > deadline:
            break
        search(3, max_d)
        if best_max_den <= max_d:
            break
    
    # Also try 4 terms with smaller max_den
    for max_d in range(best_max_den - 1, 50, -1):
        if time.time() > deadline:
            break
        search(4, max_d)
        if best_max_den <= max_d:
            break
    
    # Try 5 terms
    for max_d in range(best_max_den - 1, 50, -1):
        if time.time() > deadline:
            break
        search(5, max_d)
        if best_max_den <= max_d:
            break
    
    return best_solution
# EVOLVE-BLOCK-END
