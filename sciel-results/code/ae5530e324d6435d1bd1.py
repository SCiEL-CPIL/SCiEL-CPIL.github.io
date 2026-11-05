import time
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s, we want to finish by 60s.
    deadline = start_time + 50.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two halves
    mid = n // 2
    left_indices = list(range(mid))
    right_indices = list(range(mid, n))
    
    # Precompute fractions
    fracs = [Fraction(1, d) for d in denominators]
    
    # To improve pruning, sort the indices in each half by descending fraction value
    # i.e., ascending denominator
    left_indices.sort(key=lambda i: denominators[i])
    right_indices.sort(key=lambda i: denominators[i])
    
    def gen_subsets(indices, fracs):
        res = []
        # Using a stack for iterative DFS to avoid recursion limit/overhead?
        # Or just recursion. Depth is ~23.
        # Let's use recursion for clarity, but optimize local vars.
        
        # We need to map local index in 'indices' to global index in 'denominators'
        # But for the sum calculation, we just need the fraction values.
        # For the mask, we need to know which global index was picked.
        
        # Let's store (sum, global_mask)
        # global_mask is a bitmask of the original indices.
        
        def recurse(pos, current_sum, current_mask):
            if time.time() > deadline:
                return
            if pos == len(indices):
                if current_sum < 1:
                    res.append((current_sum, current_mask))
                return
            
            idx = indices[pos]
            val = fracs[idx]
            
            # Option 1: Exclude
            recurse(pos + 1, current_sum, current_mask)
            
            # Option 2: Include
            new_sum = current_sum + val
            if new_sum < 1:
                recurse(pos + 1, new_sum, current_mask | (1 << idx))
            # If new_sum >= 1, we don't recurse, as further additions will only increase sum
        
        recurse(0, Fraction(0), 0)
        return res

    left_subsets = gen_subsets(left_indices, fracs)
    if time.time() > deadline:
        # Fallback: Greedy
        selected, total = [], Fraction()
        for i, d in enumerate(range(2, 49)):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    right_subsets = gen_subsets(right_indices, fracs)
    
    if time.time() > deadline:
        # Fallback: Greedy
        selected, total = [], Fraction()
        for i, d in enumerate(range(2, 49)):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    # Sort right subsets by sum
    right_subsets.sort(key=lambda x: x[0])
    right_sums = [s[0] for s in right_subsets]
    
    best_sum = Fraction(0)
    best_mask = 0
    
    for l_sum, l_mask in left_subsets:
        if time.time() > deadline:
            break
        # Find max r_sum such that l_sum + r_sum < 1
        target = 1 - l_sum
        # bisect_left finds first index where right_sums[idx] >= target
        idx = bisect.bisect_left(right_sums, target)
        if idx > 0:
            r_sum, r_mask = right_subsets[idx - 1]
            total_sum = l_sum + r_sum
            if total_sum < 1 and total_sum > best_sum:
                best_sum = total_sum
                best_mask = l_mask | r_mask
    
    # Convert mask to indices
    final_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            final_indices.append(i)
            
    # Ensure non-empty
    if not final_indices:
        final_indices = [0]
        
    return final_indices
