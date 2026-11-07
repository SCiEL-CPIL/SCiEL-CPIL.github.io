"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 256
    # Generate all integers 1..256
    candidates = list(range(1, N + 1))
    
    # Precompute prime factors for each number
    # For each number, store its set of prime factors
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    # Precompute factor sets
    factor_sets = {}
    for i in candidates:
        if i == 1:
            factor_sets[i] = set()
        else:
            factor_sets[i] = prime_factors(i)
    
    # Sort candidates by value descending (heuristic: try larger values first)
    sorted_candidates = sorted(candidates, key=lambda x: -x)
    
    # We want to find a subset S of candidates such that:
    # 1. All elements are distinct (guaranteed since we pick from a set)
    # 2. Every pair has gcd 1
    # 3. Maximize sum
    
    # This is a maximum weight independent set in a conflict graph where
    # two nodes conflict if gcd > 1.
    
    # Use branch and bound with bitsets for efficiency
    n = len(candidates)
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build conflict bitset: conflict[i] is a bitmask of indices that conflict with i
    # Actually, let's use a different approach: for each candidate, track which primes it has
    # Two candidates conflict if they share any prime factor
    
    # For each prime, track which candidates have that prime
    prime_to_candidates = {}
    for v in candidates:
        for p in factor_sets[v]:
            if p not in prime_to_candidates:
                prime_to_candidates[p] = []
            prime_to_candidates[p].append(v)
    
    # Conflict: candidate i conflicts with candidate j if they share a prime
    # Let's build adjacency as sets of conflicting values
    conflict_with = {v: set() for v in candidates}
    for v in candidates:
        for p in factor_sets[v]:
            for w in prime_to_candidates[p]:
                if w != v:
                    conflict_with[v].add(w)
    
    # Now use branch and bound
    # State: current selected set (as frozenset for hashing), current sum, remaining candidates
    # We'll use a recursive approach with memoization on the "available" set
    
    # For efficiency, represent available set as a bitmask or sorted tuple
    # Since n=256, bitmask with Python ints works (arbitrary precision)
    
    # Order candidates by value descending for better pruning
    order = sorted(range(n), key=lambda i: -candidates[i])
    
    # Precompute: for each position in the order, the bitmask of candidates from that position onward
    # Actually, let's use a different representation:
    # available_mask: bitmask of which candidates are still available
    # selected_mask: bitmask of which candidates are selected
    
    # For each candidate i, precompute conflict_mask[i]: bitmask of all candidates that conflict with i
    conflict_mask = [0] * n
    for i, v in enumerate(candidates):
        for w in conflict_with[v]:
            j = val_to_idx[w]
            conflict_mask[i] |= (1 << j)
    
    # Also, candidate 1 (value 1) has no prime factors, so it conflicts with nothing
    # But we need to handle it specially - it's compatible with everything
    
    best_sum = 0
    best_selection = None
    
    def bbb(pos, available_mask, current_sum, selected_mask):
        nonlocal best_sum, best_selection
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_sum + sum of all remaining available candidates
        # This is a loose bound but fast
        remaining_sum = 0
        m = available_mask
        while m:
            # Get lowest set bit
            lb = m & (-m)
            idx = lb.bit_length() - 1
            remaining_sum += candidates[idx]
            m ^= lb
        
        if current_sum + remaining_sum <= best_sum:
            return
        
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = selected_mask
            return
        
        # Try to pick the best available candidate
        # Heuristic: pick the one with highest value that has fewest conflicts
        best_idx = -1
        best_score = -1
        m = available_mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            # Score: value - number of conflicts (prefer high value, low conflict)
            num_conflicts = bin(conflict_mask[idx] & available_mask).count('1')
            score = candidates[idx] - num_conflicts
            if score > best_score:
                best_score = score
                best_idx = idx
            m ^= lb
        
        if best_idx == -1:
            return
        
        # Option 1: Include best_idx
        new_available = available_mask & ~(1 << best_idx) & ~conflict_mask[best_idx]
        new_selected = selected_mask | (1 << best_idx)
        bbb(pos + 1, new_available, current_sum + candidates[best_idx], new_selected)
        
        # Option 2: Exclude best_idx
        new_available2 = available_mask & ~(1 << best_idx)
        bbb(pos + 1, new_available2, current_sum, selected_mask)
    
    # Start with all candidates available
    initial_mask = (1 << n) - 1
    bbb(0, initial_mask, 0, 0)
    
    # Reconstruct the solution
    if best_selection is not None:
        result = []
        m = best_selection
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            result.append(candidates[idx])
            m ^= lb
        return sorted(result)
    else:
        # Fallback: just return [1]
        return [1]

# EVOLVE-BLOCK-END
