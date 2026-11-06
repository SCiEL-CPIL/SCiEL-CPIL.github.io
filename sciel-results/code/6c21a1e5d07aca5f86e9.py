"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 30
    
    # Calculate LCM(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm(lcm_val, i)
        
    # Precompute weights
    weights = [lcm_val // (i + 1) for i in range(n)]
    
    # Meet-in-the-middle approach
    # Split 30 coefficients into two halves of 15
    # For each half, enumerate all 3^15 = 14348907 possibilities
    # Store (sum, coeffs) for first half in a dict for fast lookup
    
    half = 15
    first_half_weights = weights[:half]
    second_half_weights = weights[half:]
    
    # Generate all possible sums for the second half
    # Store in a dictionary: sum -> list of coefficient tuples (or just one if we want to save memory)
    # Since we want to find the closest to 0, we can sort the second half sums
    # and for each first half sum, binary search for the closest second half sum
    
    # To save memory and time, let's generate and store
    # Actually, 3^15 is about 14M. Storing 14M tuples of 15 ints might be heavy.
    # Let's try to be smarter.
    
    # Alternative: Generate second half sums, sort them.
    # Then for each first half sum, binary search.
    
    # Let's generate all sums for second half
    second_sums = []
    
    # Use itertools.product to generate all combinations
    # But 3^15 iterations in Python might be slow.
    # Let's see how long it takes.
    
    # Optimization: We can generate sums incrementally or use numpy if available, 
    # but standard library is safer.
    
    # Let's try to generate and store just the sums first.
    # If memory is an issue, we can be more selective.
    
    # Time check
    if time.time() > deadline - 5:
        # Fallback to simple random search if MITM is too slow to start
        pass
    
    # Generate second half sums
    # 3^15 = 14348907
    # This might take a few seconds in Python.
    
    second_sums = []
    for coeffs in product([-1, 0, 1], repeat=half):
        s = 0
        for c, w in zip(coeffs, second_half_weights):
            s += c * w
        second_sums.append(s)
    
    # Sort second sums for binary search
    second_sums.sort()
    
    # Now for each first half combination, find the best matching second half sum
    global_best_sum = None
    global_best_coeffs = None
    
    # We can also optimize by noting that if we find a sum of 1 or -1, we can stop early
    # since that's the minimum non-zero absolute value possible (in LCM units).
    
    # Iterate through first half combinations
    for coeffs1 in product([-1, 0, 1], repeat=half):
        if time.time() > deadline:
            break
        
        s1 = 0
        for c, w in zip(coeffs1, first_half_weights):
            s1 += c * w
        
        # We want s1 + s2 to be non-zero and minimize |s1 + s2|
        # So we want s2 to be close to -s1
        target = -s1
        
        # Binary search in second_sums for closest to target
        # Use bisect
        import bisect
        idx = bisect.bisect_left(second_sums, target)
        
        candidates = []
        if idx < len(second_sums):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        # Check the closest candidates
        for cand_idx in candidates:
            s2 = second_sums[cand_idx]
            total = s1 + s2
            
            if total == 0:
                continue
            
            if global_best_sum is None or abs(total) < abs(global_best_sum):
                global_best_sum = total
                # We need to recover the coefficients
                # This is the tricky part: we stored only sums, not the coefficients
                # So we need to re-search or store coefficients
                
                # To save memory, we could store (sum, coeffs) but that's heavy.
                # Alternative: Store a mapping from sum to one representative coeffs tuple.
                # But if multiple coeffs give the same sum, we might miss the optimal one?
                # No, for a given sum, any coeffs tuple that achieves it is equivalent for the sum.
                # But we need to return the coeffs.
                
                # Let's restart with storing coeffs for the second half.
                # Or, we can do a second pass to find the coeffs.
                
                # For now, let's just record the best sum and coeffs1.
                # We'll need to find the matching coeffs2 later.
                
                # This approach has a flaw: we don't have the coeffs2 readily available.
                
                pass

    # The above approach has a memory/lookup issue.
    # Let's try a different strategy:
    # 1. Generate all (sum, coeffs) for second half, store in dict: sum -> coeffs (first one found)
    # 2. For each first half, look up the target sum in the dict.
    
    # But dict with 14M entries might be too large.
    
    # Let's try a hybrid:
    # Use random search with a better heuristic, or use the MITM but with a smaller subset.
    
    # Given the time constraint, let's stick with a more efficient local search
    # but with a smarter initialization and acceptance criteria.
    
    # Actually, let's try to implement the MITM properly but efficiently.
    # We'll store the second half sums in a list, and also keep track of the coefficients
    # that produced each sum. To save memory, we can store the coefficients as a tuple
    # only for the sums we need.
    
    # Revised MITM plan:
    # 1. Generate all second half (sum, coeffs) pairs.
    # 2. Sort by sum.
    # 3. For each first half sum, binary search for the best second half sum.
    # 4. To recover coefficients, we need to have stored them.
    
    # Let's try to do this but with a time limit.
    
    # Clear the previous attempts
    global_best_sum = None
    global_best_coeffs = None
    
    # Generate second half data
    second_data = []  # list of (sum, coeffs_tuple)
    
    for coeffs in product([-1, 0, 1], repeat=half):
        if time.time() > deadline - 10:
            # Too slow, break and use fallback
            break
        s = 0
        for c, w in zip(coeffs, second_half_weights):
            s += c * w
        second_data.append((s, coeffs))
    
    if not second_data:
        # Fallback
        return [1] + [0] * 29
    
    # Sort by sum
    second_data.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    second_sums = [x[0] for x in second_data]
    
    # Now iterate through first half
    import bisect
    
    for coeffs1 in product([-1, 0, 1], repeat=half):
        if time.time() > deadline:
            break
        
        s1 = 0
        for c, w in zip(coeffs1, first_half_weights):
            s1 += c * w
        
        target = -s1
        idx = bisect.bisect_left(second_sums, target)
        
        candidates = []
        if idx < len(second_sums):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        for cand_idx in candidates:
            s2, coeffs2 = second_data[cand_idx]
            total = s1 + s2
            
            if total == 0:
                continue
            
            if global_best_sum is None or abs(total) < abs(global_best_sum):
                global_best_sum = total
                # Combine coeffs
                full_coeffs = list(coeffs1) + list(coeffs2)
                global_best_coeffs = full_coeffs
                
                if abs(total) == 1:
                    # Can't do better than 1 in LCM units
                    return global_best_coeffs

    if global_best_coeffs is None:
        return [1] + [0] * 29
        
    return global_best_coeffs
# EVOLVE-BLOCK-END
