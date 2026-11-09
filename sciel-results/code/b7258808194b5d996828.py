# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    n = 30
    
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = reduce(lcm, range(1, n + 1))
    
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    half = n // 2
    
    def enumerate_sums(start, length):
        sums = np.array([0], dtype=np.int64)
        for i in range(start, start + length):
            w = weights[i]
            # sums = np.concatenate([sums - w, sums, sums + w])
            # This creates 3 copies.
            # Better:
            new_sums = np.empty(len(sums) * 3, dtype=np.int64)
            new_sums[0:len(sums)] = sums - w
            new_sums[len(sums):2*len(sums)] = sums
            new_sums[2*len(sums):3*len(sums)] = sums + w
            sums = new_sums
        return sums

    sums1 = enumerate_sums(0, half)
    sums2 = enumerate_sums(half, n - half)
    
    sums2.sort()
    
    best_abs = np.iinfo(np.int64).max
    best_s1_val = 0
    best_s2_val = 0
    
    # Use numpy searchsorted for vectorized-ish speed? 
    # searchsorted takes an array of values.
    # We can process sums1 in chunks to avoid memory issues with large intermediate arrays if we wanted to do it all at once, 
    # but sums1 is already in memory.
    
    # targets = -sums1
    # indices = np.searchsorted(sums2, targets)
    
    # We need to check indices and indices-1.
    
    targets = -sums1
    
    # Get indices for targets
    idxs = np.searchsorted(sums2, targets)
    
    # Valid indices for sums2
    # We need to check idxs and idxs - 1
    
    # Candidate 1: sums2[idxs]
    # Be careful with bounds
    valid_idx = idxs < len(sums2)
    cands1_idx = idxs[valid_idx]
    s1_vals = sums1[valid_idx]
    s2_vals = sums2[cands1_idx]
    totals = s1_vals + s2_vals
    
    nonzero_mask = totals != 0
    if np.any(nonzero_mask):
        abs_totals = np.abs(totals[nonzero_mask])
        min_idx = np.argmin(abs_totals)
        min_val = abs_totals[min_idx]
        if min_val < best_abs:
            best_abs = min_val
            # Need to map back to original index
            # This is tricky with the mask.
            # Let's just do it in a loop if numpy is not straightforward for tracking the best pair.
            # Or, we can compute all candidates, find global min, and then find the pair.
            
            # Actually, let's just collect all candidate totals and find the min.
            # But we have two sets of candidates: idxs and idxs-1.
            
    # Let's refine the numpy approach to find the global minimum absolute non-zero sum.
    
    # Candidate set 1: pairs (s1[i], sums2[idxs[i]])
    # Candidate set 2: pairs (s1[i], sums2[idxs[i]-1])
    
    # We will create an array of all candidate totals.
    # Size of sums1 is 3^15 ~ 14M.
    # So we have ~28M candidates.
    # This might be too much memory (28M * 8 bytes = 224MB).
    # Plus the sums arrays.
    # Total memory might be high but probably okay?
    # Let's try to be smarter.
    
    # We can process in chunks.
    
    chunk_size = 100000
    best_abs = float('inf')
    best_s1_val = 0
    best_s2_val = 0
    
    for start in range(0, len(sums1), chunk_size):
        end = min(start + chunk_size, len(sums1))
        s1_chunk = sums1[start:end]
        targets_chunk = -s1_chunk
        
        idxs_chunk = np.searchsorted(sums2, targets_chunk)
        
        # Candidates from idxs
        valid = idxs_chunk < len(sums2)
        if np.any(valid):
            idxs_valid = idxs_chunk[valid]
            s1_valid = s1_chunk[valid]
            s2_valid = sums2[idxs_valid]
            totals = s1_valid + s2_valid
            nonzero = totals != 0
            if np.any(nonzero):
                abs_totals = np.abs(totals[nonzero])
                min_val = np.min(abs_totals)
                if min_val < best_abs:
                    best_abs = min_val
                    # Find which one
                    min_idx_in_chunk = np.argmin(abs_totals)
                    # Map back to original index in s1_chunk
                    # This is complex. Let's just store the values.
                    # We need s1 and s2 values.
                    # We can just get the specific value.
                    # Actually, we just need to find ONE pair that achieves best_abs.
                    # We can store the s1 and s2 values that produced the min.
                    # But we need to be careful with the mask.
                    
                    # Let's just find the index of the min in the filtered array.
                    # And then we need to know which s1 and s2 it corresponds to.
                    # This is getting complicated.
                    
                    # Alternative: Just store the best s1 and s2 values found so far.
                    # We can iterate through the candidates in the chunk and update best.
                    # But that's a Python loop over 100k items, which is fast.
                    
                    # Let's try a simpler approach:
                    # For each chunk, compute the candidate totals.
                    # Find the min absolute non-zero total in the chunk.
                    # If it's better than global best, record the s1 and s2 values.
                    
                    # To get the s1 and s2 values:
                    # We have s1_valid and s2_valid.
                    # We need the index of the min in the `nonzero` subset.
                    # Let's create a combined array of candidates for this chunk.
                    
                    pass # Placeholder
                    
        # Candidates from idxs - 1
        valid2 = idxs_chunk > 0
        if np.any(valid2):
            idxs_valid2 = idxs_chunk[valid2] - 1
            s1_valid2 = s1_chunk[valid2]
            s2_valid2 = sums2[idxs_valid2]
            totals2 = s1_valid2 + s2_valid2
            nonzero2 = totals2 != 0
            if np.any(nonzero2):
                abs_totals2 = np.abs(totals2[nonzero2])
                min_val2 = np.min(abs_totals2)
                if min_val2 < best_abs:
                    best_abs = min_val2
                    # Record values
                    # Again, need to map back.

    # This chunking logic is getting messy for tracking the specific s1, s2.
    # Given the time constraint, maybe the pure Python loop with binary search is safer if we optimize it?
    # Or just use the pure Python loop but break early if we find 1.
    # The probability of finding 1 is high?
    # The number of possible sums is huge. The density is high.
    # It is very likely that 1 is achievable.
    # If we find 1, we break.
    
    # Let's stick to the pure Python loop but with early exit.
    # The previous pure Python loop might have been too slow if it didn't find 1 quickly.
    # But 3^15 is 14M.
    # If we sort sums2, and iterate sums1, we do 14M binary searches.
    # In Python, 14M * log(14M) ~ 14M * 24 ~ 336M ops.
    # This will likely time out in 60s.
    
    # We MUST use NumPy for the search to be fast enough.
    
    # Let's implement the NumPy search correctly.
    
    # We will process the entire sums1 array at once if memory allows.
    # 14M targets.
    # 14M indices.
    # 14M s1 values.
    # 14M s2 values.
    # 14M totals.
    # Total memory ~ 5 * 14M * 8 bytes = 560MB.
    # This is risky but might work.
    
    # Let's try to do it in one go.
    
    targets = -sums1
    idxs = np.searchsorted(sums2, targets)
    
    # We need to check idxs and idxs-1.
    # Let's create an array of candidates.
    
    # Candidate 1: idxs
    mask1 = idxs < len(sums2)
    s1_c1 = sums1[mask1]
    s2_c1 = sums2[idxs[mask1]]
    totals_c1 = s1_c1 + s2_c1
    
    # Candidate 2: idxs - 1
    mask2 = idxs > 0
    s1_c2 = sums1[mask2]
    s2_c2 = sums2[idxs[mask2] - 1]
    totals_c2 = s1_c2 + s2_c2
    
    # Combine
    # We want the min absolute non-zero total.
    
    # Filter non-zero
    nz1 = totals_c1 != 0
    if np.any(nz1):
        abs_c1 = np.abs(totals_c1[nz1])
        min_c1 = np.min(abs_c1)
    else:
        min_c1 = float('inf')
        
    nz2 = totals_c2 != 0
    if np.any(nz2):
        abs_c2 = np.abs(totals_c2[nz2])
        min_c2 = np.min(abs_c2)
    else:
        min_c2 = float('inf')
        
    if min_c1 < min_c2:
        best_abs = min_c1
        # Find the index in the filtered array
        # We need to map back to the original s1, s2
        # This is hard with the mask.
        
        # Alternative: Just find the global min value, and then search for a pair that achieves it?
        # Or, we can just pick the first one that achieves the min.
        
        # Let's find the index of the min in abs_c1.
        min_idx = np.argmin(abs_c1)
        # We need to find the corresponding index in the original arrays.
        # This is tricky because of the boolean mask.
        
        # Let's just create a list of (abs_total, s1, s2) for the candidates?
        # That's 28M tuples. Too much memory.
        
        # Let's try a different approach:
        # We know the min value `best_abs`.
        # We can now search for s1, s2 such that |s1+s2| == best_abs.
        # We can iterate through s1 and check if s2 = -s1 +/- best_abs exists in sums2.
        # This is O(N log N) again.
        
        # But we can use a set for sums2?
        # sums2 is sorted. We can use binary search.
        # But we need to do it for 14M s1 values.
        
        # Wait, if we have the `totals` array, we can find the index of the min.
        # `np.argmin` returns the index in the flattened array.
        # We can reconstruct the s1 and s2 values if we keep them.
        
        # Let's keep s1_c1, s2_c1, totals_c1.
        # Find index of min in totals_c1 (filtered).
        
        # Actually, we can just use `np.where` or similar.
        
        # Let's simplify:
        # We will find the min absolute value `best_abs`.
        # Then we will find ONE pair (s1, s2) that achieves it.
        # We can do this by checking the candidates we already computed.
        
        # If min_c1 is the global min:
        # Find index in abs_c1.
        # Map back to index in totals_c1.
        # Map back to index in s1_c1.
        
        # This is doable.
        
        # Let's implement the mapping.
        
        # For candidate 1:
        # nz1 is a boolean mask of length len(totals_c1).
        # We need the index in the original `totals_c1` array.
        # We can use `np.flatnonzero(nz1)` to get the indices in the original array.
        # Then find which of those indices has the min abs value.
        
        indices_nz1 = np.flatnonzero(nz1)
        abs_vals1 = np.abs(totals_c1[indices_nz1])
        min_idx_in_nz1 = np.argmin(abs_vals1)
        orig_idx1 = indices_nz1[min_idx_in_nz1]
        
        best_s1_val = s1_c1[orig_idx1]
        best_s2_val = s2_c1[orig_idx1]
        
    else:
        best_abs = min_c2
        indices_nz2 = np.flatnonzero(nz2)
        abs_vals2 = np.abs(totals_c2[indices_nz2])
        min_idx_in_nz2 = np.argmin(abs_vals2)
        orig_idx2 = indices_nz2[min_idx_in_nz2]
        
        best_s1_val = s1_c2[orig_idx2]
        best_s2_val = s2_c2[orig_idx2]

    # Reconstruction
    def find_coeffs(start, length, target_sum):
        coeffs = [0] * length
        suffix_sums = [0] * (length + 1)
        for i in range(length - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i+1] + weights[start + i]
            
        def dfs(pos, current_sum):
            if pos == length:
                return current_sum == target_sum
            
            w = weights[start + pos]
            diff = target_sum - current_sum
            if diff > suffix_sums[pos] or diff < -suffix_sums[pos]:
                return False
                
            for c in (-1, 0, 1):
                coeffs[pos] = c
                if dfs(pos + 1, current_sum + c * w):
                    return True
            
            coeffs[pos] = 0
            return False
            
        if dfs(0, 0):
            return coeffs
        return None

    coeffs1 = find_coeffs(0, half, best_s1_val)
    coeffs2 = find_coeffs(half, n - half, best_s2_val)
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
        
    return coeffs1 + coeffs2

# EVOLVE-BLOCK-END
