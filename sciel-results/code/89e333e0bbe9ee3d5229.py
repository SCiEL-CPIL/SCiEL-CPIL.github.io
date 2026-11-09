"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from heapq import heappush, heappop
import itertools

# EVOLVE-BLOCK-START
def construct():
    """Find minimum number of distinct unit fractions (denominators >= 5) summing to 1."""
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    
    # Use a priority queue to search for minimal solutions
    # State: (num_terms, remaining_sum, used_denoms_frozenset, current_list)
    # We want to minimize num_terms
    
    # Start with the target: sum of reciprocals = 1
    # We'll use a best-first search where we expand the most promising states first
    
    # Key insight: use the identity 1/n = 1/(n+1) + 1/(n(n+1)) to split terms
    # Start from {1} and split until all terms have denominators >= 5 and are distinct
    
    # Better approach: direct backtracking with strong pruning
    # Try k=3,4,5,... and for each, find if k distinct denominators >= 5 sum to 1
    
    def can_complete(remaining, k, min_denom, used):
        """Check if it's possible to complete with k terms from min_denom upward."""
        if k == 0:
            return remaining == 0
        if k == 1:
            # Need 1/d = remaining, d = 1/remaining must be integer >= min_denom
            if remaining <= 0:
                return False
            d = 1 / remaining
            if d < min_denom or d < 5:
                return False
            if abs(d - round(d)) > 1e-9:
                return False
            d_int = int(round(d))
            if d_int in used or d_int > 10000000:
                return False
            return True
        
        # Max possible sum with k terms starting from min_denom
        max_sum = sum(1 / (min_denom + i) for i in range(k))
        if max_sum < remaining:
            return False
        
        # Min possible: each term is at most 1/min_denom, but they must be distinct
        # Actually min sum is very small, so no lower bound issue
        return True
    
    def search(k, remaining, min_denom, used, current):
        """Find k distinct denominators >= min_denom summing to remaining."""
        if time.time() > deadline:
            return None
        
        if k == 0:
            return [] if remaining == 0 else None
        
        if k == 1:
            if remaining <= 0:
                return None
            d = 1 / remaining
            if d < min_denom or d < 5:
                return None
            if abs(d - round(d)) > 1e-9:
                return None
            d_int = int(round(d))
            if d_int in used or d_int > 10000000:
                return None
            return [d_int]
        
        # Lower bound for first denominator: 1/d <= remaining/k => d >= k/remaining
        lb = max(min_denom, int(k / remaining) + 1)
        
        # Upper bound: 1/d < remaining (strictly, since other k-1 terms are positive)
        # Also need room for k-1 more terms
        # 1/d + (k-1)/(d+1) >= remaining gives upper bound on d
        # Try: 1/d < remaining => d > 1/remaining
        ub = int(1 / remaining) + 1
        
        # Tighter: we need 1/d + sum of (k-1) smallest possible terms from d+1 >= remaining
        # The (k-1) smallest terms from d+1 are 1/(d+1), 1/(d+2), ..., 1/(d+k-1)
        # So 1/d + sum(1/(d+i) for i in range(1,k)) >= remaining
        # This gives an upper bound on d
        
        for d in range(lb, ub + 1):
            if time.time() > deadline:
                return None
            
            if d in used:
                continue
            
            frac = Fraction(1, d)
            if frac >= remaining:
                continue
            
            new_rem = remaining - frac
            
            # Check if remaining can be completed
            if not can_complete(new_rem, k - 1, d + 1, used | {d}):
                continue
            
            result = search(k - 1, new_rem, d + 1, used | {d}, current + [d])
            if result is not None:
                return current + result
        
        return None
    
    # Try increasing k
    for k in range(3, 20):
        if time.time() > deadline:
            break
        
        result = search(k, Fraction(1, 1), 5, set(), [])
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(result) == k and all(5 <= d <= 10000000 for d in result) and len(set(result)) == k:
                return result
    
    # Fallback: known solution with 25 terms
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/42 + 1/70 + 1/140 + 1/280 + 1/560 + 1/1120 + 1/2240 + 1/4480 + 1/8960 + 1/17920 + 1/35840 + 1/71680 + 1/143360 + 1/286720 + 1/573440 + 1/1146880 + 1/2293760 + 1/4587520 + 1/9175040
    fallback = [5, 6, 7, 10, 14, 21, 42, 70, 140, 280, 560, 1120, 2240, 4480, 8960, 17920, 35840, 71680, 143360, 286720, 573440, 1146880, 2293760, 4587520, 9175040]
    s = sum(Fraction(1, d) for d in fallback)
    if s == 1:
        return fallback
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
