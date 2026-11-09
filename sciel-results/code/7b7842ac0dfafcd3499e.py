import numpy as np
from math import gcd

def construct():
    # Calculate LCM of 1 to 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
        
    # Weights
    weights = np.array([L // i for i in range(1, 31)], dtype=np.int64)
    
    # Split into two halves
    w1 = weights[:15]
    w2 = weights[15:]
    
    def generate_sums(weights):
        sums = np.array([0], dtype=np.int64)
        idxs = np.array([0], dtype=np.int64)
        power = 1
        for w in weights:
            # New sums: s - w, s, s + w
            # New idxs: idx - power, idx, idx + power
            new_sums = np.concatenate([sums - w, sums, sums + w])
            new_idxs = np.concatenate([idxs - power, idxs, idxs + power])
            sums = new_sums
            idxs = new_idxs
            power *= 3
        return sums, idxs

    sums1, idxs1 = generate_sums(w1)
    sums2, idxs2 = generate_sums(w2)
    
    # Sort sums2 and keep idxs2 aligned
    sort_idx = np.argsort(sums2)
    sums2_sorted = sums2[sort_idx]
    idxs2_sorted = idxs2[sort_idx]
    
    # Find best match
    # We want sums1[i] + sums2[j] to be close to 0 but not 0
    # Use searchsorted on -sums1
    neg_sums1 = -sums1
    pos = np.searchsorted(sums2_sorted, neg_sums1)
    
    # Handle boundaries
    pos_left = np.maximum(pos - 1, 0)
    pos_right = np.minimum(pos, len(sums2_sorted) - 1)
    
    # Get candidate sums from sums2
    cands_left = sums2_sorted[pos_left]
    cands_right = sums2_sorted[pos_right]
    
    # Compute total sums
    totals_left = sums1 + cands_left
    totals_right = sums1 + cands_right
    
    # We need to find the minimum non-zero absolute value
    # We can combine the candidates and filter
    # To avoid creating huge intermediate arrays, we can check in chunks or just do it all at once
    # 3^15 is ~14M, so 28M integers is fine.
    
    all_totals = np.concatenate([totals_left, totals_right])
    all_idxs1 = np.concatenate([idxs1, idxs1]) # Wait, idxs1 needs to be repeated
    # Actually, we need to keep track of which idx1 went with which idx2
    # Let's just find the min value first, then recover.
    
    # Filter non-zero
    non_zero_mask = all_totals != 0
    if not np.any(non_zero_mask):
        # Fallback, though unlikely
        return [0] * 28 + [1, -1]
        
    non_zero_totals = all_totals[non_zero_mask]
    
    # Find index of minimum absolute value
    abs_totals = np.abs(non_zero_totals)
    min_idx_in_nonzero = np.argmin(abs_totals)
    min_val = non_zero_totals[min_idx_in_nonzero]
    
    # Now we need to recover the coefficients for this specific min_val
    # We need to find i, j such that sums1[i] + sums2[j] == min_val
    # And that this pair was part of the candidates that produced min_val
    
    # We can just search for it again
    # Iterate through sums1
    # This might be slow if done in Python loop, but we only need to find ONE pair
    # We can use a set for sums2 for O(1) lookup?
    # Set of 14M integers is okay.
    
    # But we also need the indices.
    # Let's just use the sorted array and binary search for each sums1 element?
    # No, we can just check the candidates we already generated?
    # The min_val came from either totals_left or totals_right.
    # We can find which one it was.
    
    # Actually, simpler:
    # Create a dictionary for sums2: sum -> index in sorted array
    # This is O(N) memory and time.
    
    # Or, since we have the sorted array, we can just do a linear scan of sums1?
    # 14M iterations in Python is ~1-2 seconds. Acceptable.
    
    target = min_val
    found_i = -1
    found_j = -1
    
    # We can vectorize the check for the target sum
    # We want sums1[i] + sums2[j] == target
    # => sums2[j] == target - sums1[i]
    
    # Let's just use the set approach for speed and simplicity of recovery
    # Map sum value to index in sorted array
    # If there are duplicates, any index is fine.
    
    # Constructing a dict of 14M entries might be slow/memory heavy.
    # Let's stick to the sorted array and bisect.
    # But we need to do it for each i?
    
    # Alternative:
    # We know min_val is in all_totals.
    # We can find the position of min_val in all_totals?
    # No, all_totals is just values.
    
    # Let's just iterate through sums1 and use bisect on sums2_sorted.
    # We stop as soon as we find a match.
    
    import bisect
    
    for i in range(len(sums1)):
        s1 = sums1[i]
        s2_needed = target - s1
        # Find s2_needed in sums2_sorted
        j = bisect.bisect_left(sums2_sorted, s2_needed)
        if j < len(sums2_sorted) and sums2_sorted[j] == s2_needed:
            found_i = i
            found_j = j
            break
        # Also check j-1 in case of duplicates or off-by-one?
        # bisect_left finds the first occurrence. If it exists, it's at j.
        
    if found_i == -1:
        # Should not happen
        return [0] * 28 + [1, -1]
        
    idx1 = idxs1[found_i]
    idx2 = idxs2_sorted[found_j]
    
    # Decode idx1 and idx2 to coefficients
    def decode_idx(idx, n):
        coeffs = []
        for k in range(n):
            digit = idx % 3
            # digit 0 -> -1, 1 -> 0, 2 -> 1
            # Wait, my encoding was:
            # -1: idx - 3^k
            # 0: idx
            # 1: idx + 3^k
            # So if we look at idx, the digit at position k is:
            # If we added 3^k, digit is 1?
            # Let's re-verify the encoding.
            # Start idx=0.
            # Step 1:
            # -1: idx - 1 = -1. (-1 % 3 = 2).
            # 0: idx = 0. (0 % 3 = 0).
            # 1: idx + 1 = 1. (1 % 3 = 1).
            # So digit 2 -> -1, digit 0 -> 0, digit 1 -> 1.
            
            # Let's check:
            # If coeff is -1, we subtract 3^k.
            # If coeff is 0, we add 0.
            # If coeff is 1, we add 3^k.
            
            # So the base-3 digits of the final index directly map to the coefficients?
            # Let's test with 2 digits.
            # c1=1, c2=-1.
            # Start 0.
            # Add 3^0 = 1. idx=1.
            # Subtract 3^1 = 3. idx = 1 - 3 = -2.
            # -2 in base 3?
            # Python's % is non-negative.
            # -2 % 3 = 1.
            # -2 // 3 = -1.
            # Next digit: -1 % 3 = 2.
            # So digits are 1, 2.
            # Mapping: 1->1, 2->-1.
            # So c1=1, c2=-1. Correct.
            
            c = digit
            if c == 0:
                coeffs.append(0)
            elif c == 1:
                coeffs.append(1)
            else: # c == 2
                coeffs.append(-1)
            idx = idx // 3
        return coeffs
        
    coeffs1 = decode_idx(idx1, 15)
    coeffs2 = decode_idx(idx2, 15)
    
    return coeffs1 + coeffs2

if __name__ == "__main__":
    construct()
