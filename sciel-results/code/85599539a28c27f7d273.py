"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from math import gcd

def construct():
    deadline = time.time() + 85
    
    target_num = 5
    target_den = 31
    
    # Best known solution
    best = [8, 31, 248]
    best_max = 248
    
    def verify(denoms):
        total = 0
        for d in denoms:
            total += 1.0 / d
        # Use exact fraction check
        from fractions import Fraction
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        return s == Fraction(target_num, target_den)
    
    # Dijkstra-like search: state is (max_denom_so_far, remainder_num, remainder_den, path)
    # We want to minimize max_denom_so_far when remainder becomes 0
    
    # Start: remainder = 5/31, max_denom = 0, path = []
    # We'll use a priority queue ordered by max_denom_so_far
    
    # To avoid infinite search, we bound the number of terms and denominators
    MAX_TERMS = 8
    MAX_DENOM_LIMIT = best_max - 1  # We're trying to beat 248
    
    # State: (max_denom, num, den, path_tuple)
    # We reduce fractions to canonical form
    
    # Initial state
    start_state = (0, target_num, target_den, ())
    
    # Priority queue: (max_denom, num, den, path)
    # We want to explore states with smaller max_denom first
    pq = [start_state]
    
    # Visited states: (num, den) -> best max_denom seen
    visited = {}
    visited[(target_num, target_den)] = 0
    
    # For each state, we can subtract 1/k for k > last_denom in path
    # The remainder becomes (num*k - den) / (den*k)
    
    while pq and time.time() < deadline:
        max_d, num, den, path = heapq.heappop(pq)
        
        # Check if this is a better solution
        if num == 0:
            if max_d < best_max:
                best_max = max_d
                best = list(path)
            continue
        
        # Skip if we've seen this state with a better or equal max_denom
        if (num, den) in visited and visited[(num, den)] <= max_d:
            continue
        
        visited[(num, den)] = max_d
        
        # If we've used too many terms, skip
        if len(path) >= MAX_TERMS:
            continue
        
        # Determine the minimum next denominator
        # Must be > last denominator in path (to ensure distinct and increasing)
        if path:
            min_next = path[-1] + 1
        else:
            min_next = 1
        
        # The next denominator k must satisfy: 1/k < num/den (so remainder stays positive)
        # i.e., k > den/num
        # Also k >= min_next
        # And k <= MAX_DENOM_LIMIT (we're trying to find something < best_max)
        # Actually, k can be up to best_max - 1 since we want to beat best_max
        
        # Upper bound on k: we need num/den - 1/k > 0, so k > den/num
        # Lower bound: k >= min_next
        # We also need the remainder to be expressible, but let's just enumerate
        
        # k range: [min_next, MAX_DENOM_LIMIT]
        # But we need num/den > 1/k, i.e., k > den/num
        k_min = max(min_next, den // num + 1)
        k_max = MAX_DENOM_LIMIT
        
        if k_min > k_max:
            continue
        
        # To keep the search manageable, we limit the range of k we explore
        # from each state. We'll explore k in a bounded window.
        # For efficiency, we only explore a limited number of k values per state.
        
        # Actually, let's be smarter: for a given remainder num/den, the next term 1/k
        # should be "close" to num/den to keep the remainder manageable.
        # The ideal k is around den/num.
        
        ideal_k = den // num if num > 0 else k_max
        
        # Explore k in a range around ideal_k, but at least from k_min
        # We'll explore k from k_min to min(k_max, ideal_k + 200) to bound the search
        
        k_upper = min(k_max, max(k_min, ideal_k + 300))
        
        for k in range(k_min, k_upper + 1):
            if time.time() > deadline:
                break
            
            # New remainder: num/den - 1/k = (num*k - den) / (den*k)
            new_num = num * k - den
            new_den = den * k
            
            if new_num < 0:
                # We went too far, k is too small
                continue
            
            if new_num == 0:
                # Perfect! We've reached zero
                new_max_d = max(max_d, k)
                if new_max_d < best_max:
                    best_max = new_max_d
                    best = list(path) + [k]
                continue
            
            # Reduce the fraction
            g = gcd(new_num, new_den)
            red_num = new_num // g
            red_den = new_den // g
            
            # Check if we've visited this state with a better max_denom
            new_max_d = max(max_d, k)
            if (red_num, red_den) in visited and visited[(red_num, red_den)] <= new_max_d:
                continue
            
            # Only push if new_max_d < best_max (we can still beat the current best)
            if new_max_d >= best_max:
                continue
            
            new_path = path + (k,)
            heapq.heappush(pq, (new_max_d, red_num, red_den, new_path))
        
        if time.time() > deadline:
            break
    
    # Final verification
    if verify(best):
        return best
    else:
        # Fallback to known good solution
        return [8, 31, 248]
    
# EVOLVE-BLOCK-END
