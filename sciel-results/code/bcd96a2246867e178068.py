import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # We need to find n in [1000, 1000000] such that the window [n+1, n+24]
    # has the minimum number of distinct prime factors from the list above.
    
    # The numbers involved are from 1001 to 1000024.
    # Let's create an array for all numbers in this range.
    min_num = 1001
    max_num = 1000024
    total_nums = max_num - min_num + 1
    
    # signatures[i] will be a bitmask of primes dividing (min_num + i)
    signatures = np.zeros(total_nums, dtype=np.int32)
    
    # For each prime, mark its multiples in the signatures array
    for p in primes:
        # Find the first multiple of p >= min_num
        first_multiple = ((min_num + p - 1) // p) * p
        # Indices in signatures array for these multiples
        # index = multiple - min_num
        indices = np.arange(first_multiple - min_num, total_nums, p)
        signatures[indices] |= (1 << primes.index(p))
        
    # Now we need to compute the bitwise OR of signatures for windows of size 24.
    # A window starting at index j in signatures corresponds to numbers:
    # min_num + j ... min_num + j + 23
    # These numbers are n+1 ... n+24.
    # So n = min_num + j - 1.
    # We require n in [1000, 1000000].
    # 1000 <= min_num + j - 1 <= 1000000
    # 1000 <= 1001 + j - 1 <= 1000000
    # 1000 <= 1000 + j <= 1000000
    # 0 <= j <= 999000
    
    # The window uses signatures[j ... j+23].
    # So j must be such that j+23 < total_nums.
    # total_nums = 1000024 - 1001 + 1 = 1000024.
    # j+23 <= 1000023 => j <= 1000000.
    # This matches the upper bound for n.
    
    # We want to compute the OR for all valid j in [0, 999000].
    # We can do this by summing the "presence" of each prime bit in the window.
    # If a bit is present in the OR, the sum of that bit across the 24 numbers is > 0.
    # Instead of bitwise OR which is hard to vectorize for sliding windows directly without loops,
    # we can compute for each prime p, a boolean array of where it appears, then sum over the window.
    # But we have 9 primes. We can compute 9 sliding sums (using cumsum) and then count how many are > 0.
    
    # Let's use the cumsum trick for each prime bit.
    # bit_arrays[p_idx] is a boolean array of length total_nums where True if signatures[i] has bit p_idx set.
    
    # To save memory and time, we can process the windows in chunks or just compute the final count array.
    # Let's compute an array `distinct_counts` of length 999001 (for j=0..999000).
    
    # Actually, we can just loop over the 9 primes, compute the sliding sum of presence, 
    # and add 1 to distinct_counts where the sum > 0.
    
    num_windows = 999001 # j from 0 to 999000 inclusive
    distinct_counts = np.zeros(num_windows, dtype=np.int8)
    
    for p_idx, p in enumerate(primes):
        # Create boolean array for this prime
        # We already have signatures, but extracting bits is fast.
        # Or we can re-derive the mask. Re-deriving is safer/faster than bit ops on int32 array?
        # Let's just use the signatures array we built.
        mask = (signatures >> p_idx) & 1
        
        # Compute sliding window sum of mask for window size 24.
        # cumsum(mask) gives prefix sums.
        # sum(j..j+23) = cumsum[j+24] - cumsum[j]
        # We need to be careful with boundaries.
        
        # Pad mask with zeros at the end to handle the last windows easily?
        # Or just compute cumsum and slice.
        cumsum = np.cumsum(mask, dtype=np.int32)
        
        # We want sums for windows starting at j=0..999000.
        # The window covers indices j to j+23.
        # Sum = cumsum[j+24] - cumsum[j]
        
        # We need cumsum indices from 24 to 999000+24 = 1000024.
        # cumsum has length total_nums = 1000024.
        # Indices 0..1000023.
        # So cumsum[24]..cumsum[1000024] is out of bounds for the last one?
        # cumsum[i] is sum of mask[0..i-1].
        # Sum of mask[j..j+23] is cumsum[j+24] - cumsum[j].
        # Max j = 999000.
        # Max index needed in cumsum: 999000 + 24 = 1000024.
        # But cumsum length is 1000024, so max index is 1000023.
        # We need cumsum[1000024] which is sum of all elements.
        # Let's append a 0 to cumsum or just handle the last element separately?
        # Easier: append a zero to the mask before cumsum? No, that shifts indices.
        # Let's just compute cumsum of length total_nums+1.
        
        # Actually, standard trick:
        # padded_mask = np.concatenate([mask, np.zeros(24, dtype=np.int32)])
        # This makes sliding sum easier?
        
        # Let's stick to the subtraction.
        # We need values at indices 24..1000024.
        # We have cumsum[0..1000023].
        # We can create a new array `sliding_sums` of length num_windows.
        # sliding_sums[j] = cumsum[j+24] - cumsum[j]
        
        # To avoid out-of-bounds, we can note that for j up to 999000:
        # j+24 goes up to 1000024.
        # Since cumsum is length 1000024, index 1000024 is invalid.
        # We can just compute the total sum of the array once.
        total_sum = cumsum[-1]
        
        # For j where j+24 == total_nums (i.e., j=999000), the sum is total_sum - cumsum[999000].
        # For other j, it is cumsum[j+24] - cumsum[j].
        
        # Let's construct the array carefully.
        # indices for cumsum upper bound: 24 to 1000023 (for j=0 to 999000-1=998999)
        # Wait, j goes up to 999000.
        # For j=999000, upper index is 1000024.
        
        # Let's just use a simpler method:
        # Create an array of the sliding sums using numpy's stride tricks or just a loop?
        # A loop over 1M elements in Python is too slow.
        # But we can vectorize the subtraction if we pad.
        
        # Let's pad the mask with 24 zeros at the end.
        # padded_mask = np.concatenate([mask, np.zeros(24, dtype=np.int32)])
        # Now length is 1000024 + 24 = 1000048.
        # cumsum_padded = np.cumsum(padded_mask)
        # sliding_sum[j] = cumsum_padded[j+24] - cumsum_padded[j]
        # This works for all j in 0..999000 because j+24 <= 1000024 < 1000048.
        
        padded_mask = np.concatenate([mask, np.zeros(24, dtype=np.int32)])
        cumsum_padded = np.cumsum(padded_mask, dtype=np.int32)
        
        # Extract sliding sums
        # j ranges from 0 to 999000
        # upper indices: 24 to 1000024
        # lower indices: 0 to 999000
        sliding_sums = cumsum_padded[24:24+num_windows] - cumsum_padded[0:num_windows]
        
        # If sliding_sums[j] > 0, then prime p is present in the window.
        # Add 1 to distinct_counts where sliding_sums > 0.
        distinct_counts[sliding_sums > 0] += 1
        
        if time.time() > deadline:
            break

    # Find the minimum distinct count and the corresponding index j
    min_count = np.min(distinct_counts)
    
    # Get all indices with this minimum count
    candidates = np.where(distinct_counts == min_count)[0]
    
    if len(candidates) == 0:
        # Fallback
        return 1000
        
    # Pick the first one (or any)
    best_j = candidates[0]
    
    # n = min_num + best_j - 1
    best_n = min_num + best_j - 1
    
    # Verify n is in range
    if best_n < 1000:
        best_n = 1000
    if best_n > 1000000:
        best_n = 1000000
        
    return best_n
