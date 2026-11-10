import numpy as np
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    
    # Denominators 2 to 48
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Split into two halves for Meet-in-the-Middle
    # 23 and 24 elements
    n1 = 23
    n2 = n - n1
    d1 = denoms[:n1]
    d2 = denoms[n1:]
    
    # Helper to generate all subset sums and masks
    def generate_sums_masks(d_list, n_elems):
        sums = np.array([0.0], dtype=np.float64)
        masks = np.array([0], dtype=np.int32)
        
        for i in range(n_elems):
            val = 1.0 / d_list[i]
            # Create new sums and masks
            new_sums = sums + val
            new_masks = masks | (1 << i)
            
            # Concatenate
            sums = np.concatenate([sums, new_sums])
            masks = np.concatenate([masks, new_masks])
            
        return sums, masks

    # Generate for first half
    sums1, masks1 = generate_sums_masks(d1, n1)
    
    # Generate for second half
    sums2, masks2 = generate_sums_masks(d2, n2)
    
    # Sort sums2 and reorder masks2 accordingly
    sort_idx = np.argsort(sums2)
    sums2 = sums2[sort_idx]
    masks2 = masks2[sort_idx]
    
    # For each sum in sums1, find the best sum in sums2 such that sum1 + sum2 < 1
    # We want to maximize sum1 + sum2, which is equivalent to minimizing 1 - (sum1 + sum2)
    
    # targets = 1.0 - sums1
    # We need the largest element in sums2 that is strictly less than target
    # np.searchsorted(sums2, target, side='left') gives the index of the first element >= target
    # So the index of the largest element < target is pos - 1
    
    targets = 1.0 - sums1
    pos = np.searchsorted(sums2, targets, side='left') - 1
    
    # Valid indices are those where pos >= 0
    valid = pos >= 0
    
    if not np.any(valid):
        # Fallback: return a simple valid subset
        # e.g., index 0 (denom 2) is 0.5 < 1
        return [0]
        
    valid_indices = np.where(valid)[0]
    
    # Calculate candidate sums
    # sums1[valid_indices] + sums2[pos[valid_indices]]
    candidate_sums = sums1[valid_indices] + sums2[pos[valid_indices]]
    
    # We want the candidates with the largest sums (smallest deficits)
    # Sort candidate_sums in descending order
    # To get top K, we can use argpartition or just sort if K is small relative to N
    # N is approx 2^23 = 8M. Sorting 8M elements is fast.
    # But we only need top 100.
    
    # Get indices of the 100 largest sums
    K = 100
    if len(candidate_sums) > K:
        # argpartition returns indices that would sort the array, but only the first K are sorted
        top_k_indices = np.argpartition(candidate_sums, -K)[-K:]
    else:
        top_k_indices = np.arange(len(candidate_sums))
        
    # Retrieve the actual indices and masks for these top candidates
    # We need to map back to the original indices in sums1 and sums2
    # top_k_indices are indices into the valid_indices array
    
    best_indices = None
    best_deficit = None
    
    # Iterate through the top K candidates (in no specific order, but we check all)
    for idx in top_k_indices:
        # idx is an index into valid_indices
        orig_idx1 = valid_indices[idx]
        orig_idx2 = pos[orig_idx1]
        
        m1 = masks1[orig_idx1]
        m2 = masks2[orig_idx2]
        
        # Reconstruct the list of indices
        indices = []
        for k in range(n1):
            if m1 & (1 << k):
                indices.append(k)
        for k in range(n2):
            if m2 & (1 << k):
                indices.append(n1 + k)
                
        if not indices:
            continue
            
        # Verify with exact rational arithmetic
        sum_frac = Fraction(0, 1)
        for i in indices:
            sum_frac += Fraction(1, denoms[i])
            
        if sum_frac < 1:
            deficit = 1 - sum_frac
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_indices = indices
                
    if best_indices is None:
        # If no valid candidate found in top K (unlikely), fall back to a greedy or simple search
        # For safety, return a known valid subset
        return [0]
        
    return sorted(best_indices)
