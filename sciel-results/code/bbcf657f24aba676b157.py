"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """Find minimum number of distinct unit fractions (denominators >= 5) summing to 1."""
    deadline = time.time() + 85  # 85s safety margin before 90s kill
    
    # Try to find solutions with increasing number of terms k = 3, 4, 5, ...
    # For each k, use backtracking search with pruning.
    
    def search_k(k, used_denoms, remaining, min_denom, current_list):
        """
        Find k distinct denominators >= 5, all >= min_denom, summing to `remaining`.
        Returns list of denominators or None.
        """
        if time.time() > deadline:
            return None
        
        if k == 0:
            return [] if remaining == 0 else None
        
        if k == 1:
            # Need 1/d = remaining, so d = 1/remaining
            if remaining > 0:
                d = 1 / remaining
                # Check if d is a positive integer >= min_denom and >= 5
                if d >= min_denom and d >= 5 and abs(d - round(d)) < 1e-9:
                    d_int = int(round(d))
                    if d_int not in used_denoms and d_int <= 10000000:
                        return [d_int]
            return None
        
        # Pruning: the maximum possible sum with k terms starting from min_denom
        # is achieved by taking the smallest possible denominators
        max_possible = sum(1 / (min_denom + i) for i in range(k))
        if max_possible < remaining:
            return None
        
        # Pruning: the minimum possible sum with k terms (largest denominators)
        # is very small, so we mainly need to check if remaining is achievable
        # Lower bound on each denominator: 1/d <= remaining/k => d >= k/remaining
        # But we also need d >= min_denom
        lower_bound = max(min_denom, int(k / remaining) + 1)
        
        # Upper bound on first denominator: 1/d1 < remaining (need room for rest)
        # 1/d1 <= remaining - sum of smallest (k-1) terms from d1+1
        # Simpler: 1/d1 < remaining, so d1 > 1/remaining
        # Also 1/d1 + (k-1)/d1 > remaining is not quite right
        # Better: 1/d1 < remaining and 1/d1 + (k-1)/(d1+1) >= remaining (optimistic)
        # Actually: 1/d1 must be < remaining (strictly, since other terms are positive)
        # And 1/d1 + (k-1)/(d1+1) >= remaining (pessimistic bound - max remaining sum)
        
        upper_bound = int(1 / remaining)  # d1 must satisfy 1/d1 < remaining => d1 > 1/remaining
        # Actually 1/d1 < remaining means d1 > 1/remaining, so d1 >= floor(1/remaining) + 1
        upper_bound = max(upper_bound, lower_bound)
        
        # Tighter upper bound: we need 1/d1 + sum of (k-1) terms each >= 1/(d1 + something) 
        # The smallest remaining sum is when we take the largest denominators, but that's unbounded.
        # Instead: 1/d1 must be less than remaining, and the remaining (k-1) terms must sum to remaining - 1/d1.
        # Each of those terms is at most 1/(d1+1), so (k-1)/(d1+1) >= remaining - 1/d1
        # => (k-1)/(d1+1) + 1/d1 >= remaining
        # This gives an upper bound on d1.
        
        best_result = None
        for d1 in range(lower_bound, upper_bound + 1):
            if time.time() > deadline:
                return best_result
            
            if d1 in used_denoms:
                continue
            
            frac = Fraction(1, d1)
            if frac >= remaining:
                continue  # 1/d1 must be strictly less than remaining
            
            new_remaining = remaining - frac
            if new_remaining <= 0:
                continue
            
            # Check if new_remaining can be achieved with k-1 terms from d1+1
            # Max sum of k-1 terms from d1+1: sum(1/(d1+1+i) for i in range(k-1))
            if k - 1 > 0:
                max_rest = sum(Fraction(1, d1 + 1 + i) for i in range(k - 1))
                if max_rest < new_remaining:
                    continue
            
            result = search_k(k - 1, used_denoms | {d1}, new_remaining, d1 + 1, current_list + [d1])
            if result is not None:
                return current_list + result
        
        return best_result
    
    # Try k = 3, 4, 5, ... up to some reasonable limit
    for k in range(3, 15):
        if time.time() > deadline:
            break
        
        # Initial remaining is 1, no used denominators, min_denom = 5
        result = search_k(k, set(), Fraction(1, 1), 5, [])
        if result is not None:
            # Verify the solution
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(result) == k and all(5 <= d <= 10000000 for d in result) and len(set(result)) == k:
                return result
    
    # Fallback: use the original DP approach with a larger target
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if time.time() > deadline:
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    # Last resort: return a known valid solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/42 + 1/70 + 1/140 + 1/280 + 1/560 + 1/1120 + 1/2240 + 1/4480 + 1/8960 + 1/17920 + 1/35840 + 1/71680 + 1/143360 + 1/286720 + 1/573440 + 1/1146880 + 1/2293760 + 1/4587520 + 1/9175040
    # Actually, let me just use a simpler known decomposition
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + ...
    # This is getting complicated. Let me use the split identity.
    
    # Use the identity 1/n = 1/(n+1) + 1/(n(n+1)) to build a solution
    # Start with 1 = 1/1, but we need denominators >= 5
    # 1 = 1/2 + 1/2, but that's not distinct
    # Let me use: 1 = 1/3 + 1/3 + 1/3, not distinct
    # 1 = 1/2 + 1/3 + 1/6
    # Split 1/2: 1/2 = 1/3 + 1/6, but 1/6 is already there
    # 1/2 = 1/4 + 1/4, not distinct
    # 1/2 = 1/5 + 1/10
    # So 1 = 1/5 + 1/10 + 1/3 + 1/6, but 3 and 6 are < 5
    # 1/3 = 1/4 + 1/12, 4 < 5
    # 1/3 = 1/5 + 1/15
    # So 1 = 1/5 + 1/10 + 1/5 + 1/15 + 1/6, but 5 is duplicated and 6 < 5
    
    # Let me just use a larger common denominator approach
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
