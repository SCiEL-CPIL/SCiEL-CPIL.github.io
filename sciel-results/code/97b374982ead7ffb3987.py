"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    
    # Deadline guard: 90s limit, use 85s for safety
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Denominators are 2 to 48 inclusive
    # Indices 0 to 46 correspond to denominators 2 to 48
    n = 47
    denominators = list(range(2, 49))
    
    # Calculate total sum of all reciprocals to determine bit-width
    # Sum_{i=2}^{48} 1/i is approx 3.27
    # We need enough bits to represent the total sum as an integer
    # Let's use a large enough bit width, say 60 bits, to be safe for exact integer representation
    # Actually, we can just use Python's arbitrary precision integers for the bitset.
    # However, standard integers in Python are efficient for bit ops up to certain sizes.
    # 2^47 subsets is 140 trillion. This is too large for a simple loop in Python.
    
    # Wait, 2^47 is 1.4 * 10^14. A simple loop in Python will take centuries.
    # We need a smarter approach or a different strategy.
    
    # Let's re-evaluate the size.
    # If we use NumPy, can we do it?
    # NumPy arrays of 2^47 bits = 2^40 bytes = 1 TB. Not possible.
    
    # So exhaustive search over all 2^47 subsets is not feasible in 90s in Python.
    
    # Alternative Strategy:
    # The sum of 1/2 + 1/3 + ... + 1/48 is ~3.27.
    # We want to get as close to 1 as possible from below.
    # This is a subset sum problem with rational numbers.
    # Since the numbers are small (47 items), we can use dynamic programming?
    # The sum is ~3.27. If we scale by LCM(2..48), the numbers become huge.
    # LCM(2..48) is very large.
    
    # Let's look at the structure.
    # 1/2 = 0.5
    # 1/3 = 0.333
    # 1/4 = 0.25
    # 1/5 = 0.2
    # ...
    
    # Maybe a greedy approach with backtracking (Branch and Bound) is better?
    # Sort denominators by value (ascending, i.e., largest fractions first).
    # Try to include items if they don't exceed 1.
    # If we exceed, backtrack.
    # Since we want to MINIMIZE deficit, we want MAXIMIZE sum < 1.
    
    # Let's try a recursive search with pruning.
    # State: current index, current sum, best_sum_found.
    # Pruning:
    # 1. If current_sum + sum_of_remaining < best_sum_found, prune.
    # 2. If current_sum >= 1, prune (we need strictly below).
    
    # Precompute suffix sums for pruning.
    
    fractions = [Fraction(1, d) for d in denominators]
    # Sort by value descending to fill up quickly?
    # Or ascending?
    # Usually for subset sum, sorting descending helps fill the target faster,
    # but for "closest below", we might need to be careful.
    # Let's sort by value descending (largest fractions first).
    
    items = sorted([(f, i) for i, f in enumerate(fractions)], key=lambda x: -x[0])
    sorted_fracs = [x[0] for x in items]
    sorted_indices = [x[1] for x in items]
    
    # Suffix sums: sum of fractions from i to end
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_fracs[i]
    
    best_sum = Fraction(0)
    best_indices = []
    
    # To speed up, we can use a time check inside the recursion.
    # But recursion depth is 47, so overhead is small.
    
    def search(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
        
        # Pruning 1: Even if we take all remaining, can we beat best?
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # Base case: reached end of list
        if idx == n:
            if current_sum < 1 and current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices[:]
            return
        
        # Option 1: Exclude item idx
        search(idx + 1, current_sum, current_indices)
        
        # Option 2: Include item idx
        new_sum = current_sum + sorted_fracs[idx]
        if new_sum < 1:
            current_indices.append(sorted_indices[idx])
            search(idx + 1, new_sum, current_indices)
            current_indices.pop()
            
        # Optional: If we included it and got very close, we might prune more effectively,
        # but the suffix sum check handles the "can we beat" part.
        
    search(0, Fraction(0), [])
    
    # If best_indices is empty, it means no subset sum was found < 1?
    # But 1/2 < 1, so there is always at least one.
    # The initial best_sum is 0. Any valid subset has sum > 0.
    
    return best_indices
# EVOLVE-BLOCK-END
