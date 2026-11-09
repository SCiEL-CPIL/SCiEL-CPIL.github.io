"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import bisect
import time
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    start_time = time.time()
    deadline = start_time + 45.0  # 15s margin from 60s limit

    # Split into two halves
    mid = n // 2
    d1 = denominators[:mid]
    d2 = denominators[mid:]
    
    # Generate subset sums for d1 using floats
    # We store (float_sum, mask)
    # Optimization: Use a list of tuples.
    sums1 = [(0.0, 0)]
    for i, d in enumerate(d1):
        val = 1.0 / d
        # Iterate over a copy of the current list to avoid modifying while iterating
        # Or better, build a new list
        new_sums = []
        for s, m in sums1:
            new_s = s + val
            if new_s < 1.0:
                new_sums.append((new_s, m | (1 << i)))
        sums1.extend(new_sums)
        
    # Generate subset sums for d2 using floats
    sums2 = [(0.0, 0)]
    for i, d in enumerate(d2):
        val = 1.0 / d
        new_sums = []
        for s, m in sums2:
            new_s = s + val
            if new_s < 1.0:
                new_sums.append((new_s, m | (1 << i)))
        sums2.extend(new_sums)
        
    # Check deadline
    if time.time() > deadline:
        # Fallback to greedy if generation was too slow (unlikely for 23/24 split)
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    # Sort sums2 by value
    sums2.sort(key=lambda x: x[0])
    # Extract values for binary search
    vals2 = [s for s, m in sums2]
    
    # We will collect a list of candidate (deficit_float, mask1, mask2)
    candidates = []
    
    # Iterate through sums1
    # Optimization: local variable binding
    bisect_left = bisect.bisect_left
    v2 = vals2
    s2_list = sums2
    
    for s1, m1 in sums1:
        if time.time() > deadline:
            break
            
        # We need s2 < 1.0 - s1
        target = 1.0 - s1
        # Find rightmost index in vals2 such that vals2[i] < target
        # bisect_left gives first index >= target
        idx = bisect_left(v2, target)
        
        if idx > 0:
            # The best candidate is at idx - 1
            s2 = v2[idx - 1]
            m2 = s2_list[idx - 1][1]
            
            total_sum = s1 + s2
            deficit = 1.0 - total_sum
            
            candidates.append((deficit, m1, m2))
            
    # Sort candidates by deficit (ascending)
    candidates.sort(key=lambda x: x[0])
    
    best_deficit = None
    best_indices = None
    
    # Verify top candidates with Fraction
    # We check a limited number of top candidates to stay within time limits
    # If the float deficit is very small, the exact deficit is likely small too.
    # We need to ensure we don't miss a better exact solution hidden in float noise,
    # but given the discrete nature, float precision is usually sufficient for ordering
    # unless sums are extremely close. 100-200 checks should be safe.
    
    max_checks = 200
    for i in range(min(max_checks, len(candidates))):
        deficit, m1, m2 = candidates[i]
        
        # Early exit if float deficit is already worse than best exact deficit
        # (converted to float)
        if best_deficit is not None:
            if deficit > float(best_deficit) + 1e-12:
                # Since candidates are sorted by float deficit, 
                # if current float deficit is significantly larger than best exact, 
                # we might still have exact deficits smaller due to float error?
                # Actually, float deficit approximates exact deficit.
                # If float deficit > best_exact_deficit, it's *possible* exact is smaller?
                # No, usually float error is small. 
                # However, to be safe, we just check a fixed number.
                pass

        # Reconstruct indices
        indices = []
        for i_idx in range(len(d1)):
            if m1 & (1 << i_idx):
                indices.append(i_idx)
        for i_idx in range(len(d2)):
            if m2 & (1 << i_idx):
                indices.append(mid + i_idx)
                
        if not indices:
            continue
            
        # Compute exact sum
        exact_sum = Fraction(0)
        for idx in indices:
            exact_sum += Fraction(1, denominators[idx])
            
        if exact_sum < 1:
            current_deficit = 1 - exact_sum
            if best_deficit is None or current_deficit < best_deficit:
                best_deficit = current_deficit
                best_indices = indices
                
        if time.time() > deadline:
            break
            
    if best_indices is None:
        # Fallback: return a known valid small set
        # e.g. just [0] (1/2)
        best_indices = [0]
        
    return best_indices
# EVOLVE-BLOCK-END
