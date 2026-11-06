import time
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Denominators 2 to 48
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Split into two halves
    mid = n // 2
    left_denoms = denoms[:mid]
    right_denoms = denoms[mid:]
    
    # Function to generate all subset sums (float) and index masks
    def gen_subsets(d_list):
        sums = []
        masks = []
        # Use iterative method
        # Start with sum 0, mask 0
        sums.append(0.0)
        masks.append(0)
        
        for i, d in enumerate(d_list):
            val = 1.0 / d
            current_len = len(sums)
            # Append new sums
            for j in range(current_len):
                new_sum = sums[j] + val
                new_mask = masks[j] | (1 << i)
                sums.append(new_sum)
                masks.append(new_mask)
        return sums, masks

    # Generate left subsets
    left_sums, left_masks = gen_subsets(left_denoms)
    
    # Generate right subsets
    right_sums, right_masks = gen_subsets(right_denoms)
    
    # Sort right sums and keep associated masks
    # We need to sort by sum to use binary search
    # Create list of (sum, mask) and sort
    right_pairs = list(zip(right_sums, right_masks))
    right_pairs.sort(key=lambda x: x[0])
    
    # Extract sorted sums for binary search
    sorted_right_sums = [p[0] for p in right_pairs]
    # We need to map back to mask when we find an index
    # right_pairs[idx] gives (sum, mask)
    
    best_deficit = float('inf')
    best_left_idx = -1
    best_right_idx = -1
    
    # Iterate through left sums
    for i, ls in enumerate(left_sums):
        if time.time() > deadline:
            break
            
        # We need rs such that ls + rs < 1.0
        # rs < 1.0 - ls
        target = 1.0 - ls
        
        # Find the largest rs < target
        # bisect_left finds first index where sorted_right_sums[idx] >= target
        # So idx - 1 is the largest index where sum < target
        # But we must be careful with floating point precision.
        # If target is exactly a sum, we need strictly less.
        # bisect_left(target) returns insertion point to maintain order.
        # If target is 0.5 and list has 0.5, bisect_left returns index of 0.5.
        # We want strictly less, so we take index - 1.
        
        pos = bisect.bisect_left(sorted_right_sums, target)
        
        # Candidate is at pos - 1
        if pos > 0:
            rs_idx = pos - 1
            rs_val = sorted_right_sums[rs_idx]
            total = ls + rs_val
            
            # Check strictly below 1
            # Due to float precision, total might be 1.0 even if mathematically < 1.0
            # Or total might be < 1.0 but mathematically >= 1.0
            # We will verify the best candidate at the end.
            # For now, accept if total < 1.0
            
            if total < 1.0:
                deficit = 1.0 - total
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_left_idx = i
                    best_right_idx = rs_idx
                    # Optional: Early exit if deficit is extremely small?
                    # But we want global optimum.
    
    if best_left_idx == -1:
        # Fallback: return a single small denominator
        return [0] # Index 0 is denom 2, sum 0.5 < 1
    
    # Retrieve masks
    l_mask = left_masks[best_left_idx]
    r_mask = right_pairs[best_right_idx][1]
    
    # Construct indices
    # Left indices correspond to denoms 0..mid-1
    # Right indices correspond to denoms mid..n-1
    # But wait, the masks are local to the half.
    # l_mask bit i corresponds to left_denoms[i] which is denoms[i]
    # r_mask bit i corresponds to right_denoms[i] which is denoms[mid + i]
    
    indices = []
    for i in range(mid):
        if (l_mask >> i) & 1:
            indices.append(i)
            
    for i in range(len(right_denoms)):
        if (r_mask >> i) & 1:
            indices.append(mid + i)
            
    # Verification and Polish
    # The float search might have picked a candidate that is mathematically >= 1.0
    # or missed a candidate that is mathematically < 1.0 but rounded to >= 1.0 in float.
    # Let's verify the best candidate with Fraction.
    
    if time.time() < deadline - 5:
        # Verify best candidate
        s = Fraction(0, 1)
        for idx in indices:
            s += Fraction(1, denoms[idx])
            
        if s >= 1:
            # This candidate is invalid. We need to find the next best.
            # This is tricky. We might have to re-evaluate neighbors.
            # For now, if invalid, we can try to remove the smallest contribution?
            # Or just return a known valid subset like [0] (sum 0.5).
            # A better fallback: try to find a valid one from the top few candidates.
            # Let's just return [0] to be safe, or try a few other candidates.
            
            # Let's try to recover by checking the next few candidates in the right list
            # for the best left index found.
            found_valid = False
            for offset in range(1, 10):
                if best_right_idx - offset < 0: break
                alt_r_idx = best_right_idx - offset
                alt_r_mask = right_pairs[alt_r_idx][1]
                alt_indices = []
                for i in range(mid):
                    if (l_mask >> i) & 1:
                        alt_indices.append(i)
                for i in range(len(right_denoms)):
                    if (alt_r_mask >> i) & 1:
                        alt_indices.append(mid + i)
                        
                alt_s = Fraction(0, 1)
                for idx in alt_indices:
                    alt_s += Fraction(1, denoms[idx])
                    
                if alt_s < 1:
                    indices = alt_indices
                    found_valid = True
                    break
            
            if not found_valid:
                # Fallback to simple valid subset
                return [0]
                
    return indices
