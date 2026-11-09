"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from fractions import Fraction

def construct():
    """
    Find minimum number of distinct integer denominators in [5, 10^7] 
    whose reciprocal sum is exactly 1.
    """
    deadline = time.time() + 85
    
    # Verified 7-term solution: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/14
    # LCM(5,6,7,8,9,10,14) = 2520
    # 2520/5 + 2520/6 + 2520/7 + 2520/8 + 2520/9 + 2520/10 + 2520/14
    # = 504 + 420 + 360 + 315 + 280 + 252 + 180 = 2311... let me recompute
    # Actually: 504+420=924, +360=1284, +315=1599, +280=1879, +252=2131, +180=2311 != 2520
    # That's not 1. Let me find a correct one.
    
    # Known: 1 = 1/2 + 1/3 + 1/6
    # Split 1/2 = 1/5 + 1/10 (since 1/5+1/10 = 3/10 != 1/2, no)
    # 1/2 = 1/5 + 1/10 is wrong. 1/5 + 1/10 = 3/10.
    # 1/2 = 1/6 + 1/3? No, 1/6+1/3 = 1/2. But 1/3 < 5? No, 3 < 5.
    # Let me just use the A* search to find a valid solution.
    
    def verify(sol):
        if not sol:
            return False
        if len(set(sol)) != len(sol):
            return False
        if any(d < 5 or d > 10000000 for d in sol):
            return False
        s = sum(Fraction(1, d) for d in sol)
        return s == 1
    
    # A* search over remainders
    # State: (remaining_fraction, min_denominator, path_so_far)
    # Heuristic: minimum additional terms needed based on max possible sum
    
    def max_sum_from(min_d, k):
        """Maximum sum of k distinct unit fractions with denominators >= min_d."""
        # Upper bound: sum of 1/(min_d), 1/(min_d+1), ..., 1/(min_d+k-1)
        # For efficiency, use an upper bound
        if k <= 0:
            return Fraction(0, 1)
        # Conservative upper bound
        return Fraction(k, min_d)
    
    def min_terms_needed(remaining, min_d):
        """Admissible heuristic: minimum number of terms needed."""
        if remaining <= 0:
            return 0
        # Each term 1/d <= 1/min_d, so we need at least remaining * min_d terms
        # But also each term < remaining (since we need positive remainder after)
        # Better: 1/d >= remaining - (k-1)/(d+1) roughly
        # Simple lower bound: ceil(remaining * min_d)
        import math
        lb = math.ceil(remaining * min_d)
        return max(1, lb)
    
    # Priority queue: (depth, remaining, min_d, path)
    # We want to find the shortest path (fewest terms)
    
    start_remaining = Fraction(1, 1)
    start_min_d = 5
    
    # Use A*: f = g + h where g = depth, h = heuristic
    # h = min_terms_needed(remaining, min_d)
    
    open_set = []
    # (f_score, depth, remaining, min_d, path)
    h0 = min_terms_needed(start_remaining, start_min_d)
    heapq.heappush(open_set, (h0, 0, start_remaining, start_min_d, []))
    
    visited = set()  # (remaining, min_d) to avoid re-exploration
    visited.add((start_remaining, start_min_d))
    
    best_solution = None
    max_expansions = 500000
    
    expansions = 0
    while open_set and time.time() < deadline:
        f_score, depth, remaining, min_d, path = heapq.heappop(open_set)
        
        if time.time() > deadline - 5:
            break
        
        expansions += 1
        if expansions > max_expansions:
            break
        
        # If remaining is a unit fraction, we can complete
        if remaining.numerator == 1:
            d = remaining.denominator
            if d >= min_d and d >= 5 and d <= 10000000:
                sol = path + [d]
                if verify(sol):
                    best_solution = sol
                    break
        
        # Try adding 1/d for d from min_d upward
        # Lower bound for d: 1/d < remaining => d > 1/remaining
        # Upper bound: we need remaining - 1/d >= 0, and remaining - 1/d must be representable
        # Also 1/d < remaining => d > 1/remaining
        # And we need remaining - 1/d <= max_sum_from(d+1, max_terms)
        
        # Pruning: if remaining <= 0, skip
        if remaining <= 0:
            continue
        
        # d must satisfy: 1/d < remaining => d > 1/remaining
        # So d >= floor(1/remaining) + 1
        inv_rem = 1 / remaining
        d_lower = max(min_d, int(inv_rem) + 1)
        
        # Upper bound: 1/d >= remaining - (max_terms-1)/(d+1) approximately
        # For now, limit d to a reasonable range
        # The maximum d we'd consider: 1/d >= remaining/2 roughly
        # So d <= 2/remaining
        d_upper = min(10000000, int(2 / remaining) + 100)
        
        # Limit the range to avoid too many iterations
        if d_upper - d_lower > 100000:
            d_upper = d_lower + 100000
        
        for d in range(d_lower, d_upper + 1):
            if time.time() > deadline - 5:
                break
            
            frac = Fraction(1, d)
            if frac >= remaining:
                # 1/d >= remaining, so this would overshoot or equal
                # If equal, we found a solution
                if frac == remaining:
                    sol = path + [d]
                    if verify(sol):
                        if best_solution is None or len(sol) < len(best_solution):
                            best_solution = sol
                        break
                else:
                    # 1/d > remaining, skip (but d_lower should prevent this)
                    continue
            
            new_remaining = remaining - frac
            if new_remaining <= 0:
                continue
            
            new_min_d = d + 1
            
            # Check if new_remaining can be formed
            # Heuristic: min terms needed
            h = min_terms_needed(new_remaining, new_min_d)
            g = depth + 1
            f = g + h
            
            # Pruning: if f >= current best, skip
            if best_solution is not None and f >= len(best_solution):
                continue
            
            state = (new_remaining, new_min_d)
            if state in visited:
                continue
            visited.add(state)
            
            heapq.heappush(open_set, (f, g, new_remaining, new_min_d, path + [d]))
        
        if best_solution is not None:
            break
    
    if best_solution is not None and verify(best_solution):
        return best_solution
    
    # Fallback: use a verified solution
    # Let me construct one systematically
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + remainder
    # Sum so far: 1/5+1/6+1/7+1/8+1/9+1/10+1/11
    # LCM = 27720
    # 27720/5 + 27720/6 + 27720/7 + 27720/8 + 27720/9 + 27720/10 + 27720/11
    # = 5544 + 4620 + 3960 + 3465 + 3080 + 2772 + 2520 = 25961
    # 25961/27720 > 1, so this doesn't work
    
    # Let me try a different approach: use the identity 1/n = 1/(n+1) + 1/(n(n+1))
    # Start with 1 = 1/5 + 4/5
    # 4/5 = 1/6 + 19/30? No.
    
    # Use: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + ...
    # This is the harmonic series, which diverges, so we can always find a finite subset
    
    # Greedy algorithm:
    remaining = Fraction(1, 1)
    sol = []
    d = 5
    while remaining > 0 and len(sol) < 256 and time.time() < deadline:
        # Find largest d such that 1/d <= remaining and d > previous
        # 1/d <= remaining => d >= 1/remaining
        d_min = max(d, int(1/remaining) + (1 if 1/remaining == int(1/remaining) else 0))
        # Actually d >= ceil(1/remaining)
        import math
        d_min = max(d, math.ceil(1/remaining))
        
        # Use the largest d such that 1/d <= remaining
        # That's d = ceil(1/remaining)
        frac = Fraction(1, d_min)
        if frac > remaining:
            d_min += 1
            frac = Fraction(1, d_min)
        
        sol.append(d_min)
        remaining -= frac
        d = d_min + 1
        
        if remaining == 0:
            break
    
    if verify(sol):
        return sol
    
    # Last resort: return a known valid solution
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50
    # This is getting too complex. Let me just return the greedy solution if valid.
    
    raise RuntimeError("no representation found")

# EVOLVE-BLOCK-END
