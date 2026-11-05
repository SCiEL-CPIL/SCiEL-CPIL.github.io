import time
from fractions import Fraction
import numpy as np
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two halves
    n1 = n // 2
    n2 = n - n1
    
    A = denominators[:n1]
    B = denominators[n1:]
    
    # Generate subset sums and masks for A
    # We need to track the mask to reconstruct the solution
    size1 = 1 << n1
    sums_A = np.zeros(size1, dtype=np.float64)
    masks_A = np.zeros(size1, dtype=np.int32)
    
    # Efficient generation using DP
    # sums_A[i] = sum of reciprocals of bits set in i
    # We can iterate i from 1 to size1-1
    # Find least significant set bit
    # This is O(2^N) but with small constant
    
    # Note: 1<<23 is ~8M. 
    # Doing this in pure Python loop might take ~1s-2s.
    
    # To speed up, we can use the property:
    # sums_A[0] = 0
    # for i in range(1, size1):
    #   lsb = i & -i
    #   idx = lsb.bit_length() - 1
    #   prev = i ^ lsb
    #   sums_A[i] = sums_A[prev] + 1.0/A[idx]
    #   masks_A[i] = i # The mask is just i itself if we map bit k to A[k]
    
    # Actually, masks_A is just the index i. We don't need to store it separately if we just use i as the mask.
    # So we only need sums_A.
    
    vals_A = np.array([1.0/d for d in A], dtype=np.float64)
    vals_B = np.array([1.0/d for d in B], dtype=np.float64)
    
    # Generate sums for A
    sums_A = np.zeros(size1, dtype=np.float64)
    for i in range(1, size1):
        lsb = i & -i
        idx = lsb.bit_length() - 1
        prev = i ^ lsb
        sums_A[i] = sums_A[prev] + vals_A[idx]
        
    if time.time() > deadline - 10:
        # Fallback to simple greedy if generation is too slow (unlikely)
        pass

    # Generate sums for B
    size2 = 1 << n2
    sums_B = np.zeros(size2, dtype=np.float64)
    for i in range(1, size2):
        lsb = i & -i
        idx = lsb.bit_length() - 1
        prev = i ^ lsb
        sums_B[i] = sums_B[prev] + vals_B[idx]
        
    if time.time() > deadline - 10:
        pass

    # Sort B sums and keep track of original indices
    # We need to map sorted sums back to their masks
    sort_idx_B = np.argsort(sums_B)
    sorted_sums_B = sums_B[sort_idx_B]
    # sorted_masks_B = sort_idx_B (the indices in the original array)
    
    # For each sum in A, find best sum in B
    # We want sum_A + sum_B < 1.0
    # So sum_B < 1.0 - sum_A
    # Use binary search to find the largest sum_B < 1.0 - sum_A
    
    best_sum = -1.0
    best_mask_A = -1
    best_mask_B = -1
    
    # We can vectorize the target calculation
    targets = 1.0 - sums_A
    
    # np.searchsorted returns the index where the target would be inserted to maintain order
    # We want the largest value strictly less than target.
    # So we search for target, get index idx.
    # The candidate is at idx - 1.
    
    # We must be careful with floating point. 
    # If we use searchsorted, it finds the position.
    # If target is exactly equal to a sum, searchsorted with side='left' gives the index of that sum.
    # We want strictly less. So if target == sum, we want the previous one.
    # If target > sum, we want the sum.
    
    # Let's use side='left'
    # pos = np.searchsorted(sorted_sums_B, targets, side='left')
    # The best candidate index in sorted_sums_B is pos - 1.
    # We need to ensure pos - 1 >= 0.
    
    pos = np.searchsorted(sorted_sums_B, targets, side='left')
    
    # Valid indices are where pos > 0
    valid_mask = pos > 0
    
    if np.any(valid_mask):
        # For each valid i in A, the best B index is pos[i] - 1
        # We need to calculate the sum for these candidates
        # candidate_B_idx = pos[valid_mask] - 1
        # candidate_B_sums = sorted_sums_B[candidate_B_idx]
        # candidate_A_sums = sums_A[valid_mask]
        # total_sums = candidate_A_sums + candidate_B_sums
        
        # We want the max total_sum.
        # However, we must also consider the case where sum_A itself is a candidate (if we don't pick any B).
        # But B can be empty (sum 0). 0 is in sorted_sums_B at index 0 (since sums are positive).
        # If target > 0, pos will be at least 1.
        # If target <= 0, pos is 0, invalid.
        
        valid_indices = np.where(valid_mask)[0]
        
        # Get the corresponding B indices in the sorted array
        b_indices_in_sorted = pos[valid_indices] - 1
        a_indices = valid_indices
        
        # Get the sums
        a_sums = sums_A[a_indices]
        b_sums = sorted_sums_B[b_indices_in_sorted]
        
        total_sums = a_sums + b_sums
        
        # Find the max
        max_idx_in_valid = np.argmax(total_sums)
        
        # The global best candidate is at a_indices[max_idx_in_valid]
        best_a_idx = a_indices[max_idx_in_valid]
        best_b_sorted_idx = b_indices_in_sorted[max_idx_in_valid]
        
        # Map back to original mask in B
        best_b_idx = sort_idx_B[best_b_sorted_idx]
        
        best_sum = total_sums[max_idx_in_valid]
        best_mask_A = best_a_idx
        best_mask_B = best_b_idx

    # It's possible that the best solution involves only A or only B, which is covered by the above (B mask 0 or A mask 0).
    # But wait, did we cover A mask 0?
    # sums_A[0] = 0. target = 1.0.
    # searchsorted(sorted_sums_B, 1.0, side='left').
    # If there is a sum < 1.0, it will find the correct one.
    
    # If no valid mask (e.g. all sums >= 1.0? No, min sum is 0).
    # If best_mask_A is -1, something went wrong.
    
    if best_mask_A == -1:
        # Fallback
        return [0]

    # Reconstruct the solution
    # best_mask_A is an integer mask for A
    # best_mask_B is an integer mask for B
    
    indices = []
    
    # Extract bits from best_mask_A
    for i in range(n1):
        if best_mask_A & (1 << i):
            indices.append(i) # Index in global list is i
            
    # Extract bits from best_mask_B
    for i in range(n2):
        if best_mask_B & (1 << i):
            indices.append(n1 + i) # Index in global list is n1 + i
            
    # Verify with Fractions
    # We need to ensure the sum is strictly < 1
    # And it's the best one.
    # Since we found the max float sum < 1 (approximately), we should verify.
    # There might be multiple subsets with very close float sums.
    # We should check this one.
    
    s_frac = Fraction(0, 1)
    for idx in indices:
        s_frac += Fraction(1, denominators[idx])
        
    if s_frac >= 1:
        # This shouldn't happen if our float logic was sound, but if it does, 
        # we need to find the next best.
        # For now, we can try to remove the smallest element? Or just return a known valid one.
        # Let's try to find a valid one by local search from this point or greedy.
        # Given the precision of floats (1e-16) and the gap (1e-20), it is extremely unlikely 
        # that a sum < 1.0 in float is >= 1.0 in exact, UNLESS the sum is exactly 1.0 in float 
        # but we used < 1.0 check? 
        # np.searchsorted with side='left' on target 1.0-sum_A.
        # If sum_A + sum_B is exactly 1.0 in float, searchsorted might point to it?
        # If target = 0.5, and sorted_sums_B has 0.5.
        # searchsorted(0.5, side='left') returns index of 0.5.
        # We take index - 1. So we take the one BEFORE 0.5.
        # So we strictly avoid sums that are >= target in float.
        # So s_float < 1.0 is guaranteed.
        # Can s_float < 1.0 but s_exact >= 1.0?
        # Only if s_exact is very close to 1.0 from above, but rounds down to < 1.0 in float.
        # This is possible.
        # If this happens, we need to find the next best.
        
        # Simple fallback: Greedy
        greedy_indices = []
        greedy_sum = 0.0
        for i in range(n):
            new_sum = greedy_sum + 1.0/denominators[i]
            if new_sum < 1.0:
                greedy_indices.append(i)
                greedy_sum = new_sum
        
        # Verify greedy
        s_g = Fraction(0,1)
        for i in greedy_indices:
            s_g += Fraction(1, denominators[i])
        if s_g < 1:
            return greedy_indices
        else:
            return [0]
            
    return indices

if __name__ == "__main__":
    res = construct()
    pass
