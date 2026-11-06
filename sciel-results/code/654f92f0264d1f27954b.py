import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Precompute weights
    # w[i] = L / (i+1) for i in 0..29
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums(weights_half):
        """
        Generate all possible sums for a half using coefficients in {-1, 0, 1}.
        """
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            # sums_new = [s-w for s in sums] + [s for s in sums] + [s+w for s in sums]
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums

    # Generate sums for both halves
    # 3^15 is ~14.3 million
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums2 for binary search
    sums2_sorted = np.sort(sums2)
    
    len_s2 = len(sums2_sorted)
    
    # We want to find pairs (s1, s2) such that |s1 + s2| is minimized and non-zero.
    # Vectorized search
    targets = -sums1
    
    # Find insertion points
    # np.searchsorted returns the index where the element would be inserted
    indices = np.searchsorted(sums2_sorted, targets)
    
    # We need to check neighbors.
    # Create a window of indices to check.
    # indices can be 0 to len_s2.
    # We check indices - 2 to indices + 2.
    
    # To handle boundaries and vectorize, we create an array of all candidate indices
    # This might be memory heavy if we create 5 arrays of size 14M.
    # 5 * 14M * 8 bytes = 560 MB. Acceptable.
    
    # Alternatively, we can just check indices, indices-1, indices+1.
    # Let's stick to indices, indices-1, indices+1 for now.
    
    # Clip indices to [0, len_s2-1]
    # But we want to check the actual neighbors.
    
    # Let's create a list of candidate indices
    # idx_0 = indices
    # idx_1 = indices - 1
    # idx_2 = indices + 1
    
    # We need to be careful with bounds.
    # If indices is 0, indices-1 is -1.
    # We can use np.clip for the values we fetch, but we need distinct candidates.
    
    # A safer way:
    # Expand indices to a 2D array of shape (len, 3)
    # But this is also memory heavy.
    
    # Let's just do 3 separate passes or create the candidate arrays.
    
    # Candidate 1: sums2_sorted[indices]
    idx_r = np.clip(indices, 0, len_s2 - 1)
    s2_r = sums2_sorted[idx_r]
    total_r = sums1 + s2_r
    
    # Candidate 2: sums2_sorted[indices - 1]
    idx_l = np.clip(indices - 1, 0, len_s2 - 1)
    s2_l = sums2_sorted[idx_l]
    total_l = sums1 + s2_l
    
    # Candidate 3: sums2_sorted[indices + 1]
    idx_rr = np.clip(indices + 1, 0, len_s2 - 1)
    s2_rr = sums2_sorted[idx_rr]
    total_rr = sums1 + s2_rr
    
    # Combine all candidates
    # We want min |total| != 0.
    
    # Stack them
    # This creates a (3, N) array.
    all_totals = np.vstack([total_r, total_l, total_rr])
    
    # Mask zeros
    mask = all_totals != 0
    
    # Set zeros to inf
    abs_totals = np.abs(all_totals)
    abs_totals[~mask] = np.iinfo(np.int64).max
    
    # Find global minimum
    min_val = np.min(abs_totals)
    
    # If min_val is inf, something went wrong (all sums are 0? Impossible)
    # Find the index of the minimum
    # np.argmin returns a flat index
    flat_idx = np.argmin(abs_totals)
    
    # Recover which candidate and which sums1 index
    # The array is (3, N).
    # flat_idx = row * N + col
    row = flat_idx // len(sums1)
    col = flat_idx % len(sums1)
    
    s1_val = sums1[col]
    
    if row == 0:
        s2_val = s2_r[col]
        # We need the index in sums2_sorted to reconstruct?
        # No, we just need the value s2_val to reconstruct coeffs.
    elif row == 1:
        s2_val = s2_l[col]
    else:
        s2_val = s2_rr[col]
        
    # Reconstruct coefficients
    def find_coeffs(weights_half, target_sum):
        """
        Find coefficients in {-1, 0, 1} such that sum(c_i * w_i) == target_sum.
        Uses MITM for small N (15).
        """
        n_half = len(weights_half)
        mid_h = n_half // 2
        w_a = weights_half[:mid_h]
        w_b = weights_half[mid_h:]
        
        # Generate sums for w_a
        sums_a = {0: []}
        for i, w in enumerate(w_a):
            new_sums = {}
            for s, coeffs in sums_a.items():
                # -w
                ns = s - w
                new_coeffs = coeffs + [-1]
                if ns not in new_sums:
                    new_sums[ns] = new_coeffs
                # 0
                ns = s
                new_coeffs = coeffs + [0]
                if ns not in new_sums:
                    new_sums[ns] = new_coeffs
                # +w
                ns = s + w
                new_coeffs = coeffs + [1]
                if ns not in new_sums:
                    new_sums[ns] = new_coeffs
            sums_a = new_sums
            
        # Generate sums for w_b
        sums_b = {0: []}
        for i, w in enumerate(w_b):
            new_sums = {}
            for s, coeffs in sums_b.items():
                # -w
                ns = s - w
                new_coeffs = coeffs + [-1]
                if ns not in new_sums:
                    new_sums[ns] = new_coeffs
                # 0
                ns = s
                new_coeffs = coeffs + [0]
                if ns not in new_sums:
                    new_sums[ns] = new_coeffs
                # +w
                ns = s + w
                new_coeffs = coeffs + [1]
                if ns not in new_sums:
                    new_sums[ns] = new_coeffs
            sums_b = new_sums
            
        # Find match
        for s_b, coeffs_b in sums_b.items():
            target_a = target_sum - s_b
            if target_a in sums_a:
                coeffs_a = sums_a[target_a]
                return coeffs_a + coeffs_b
        
        return None # Should not happen

    coeffs1 = find_coeffs(w1, s1_val)
    coeffs2 = find_coeffs(w2, s2_val)
    
    if coeffs1 is None or coeffs2 is None:
        # Fallback to a known valid solution if reconstruction fails
        # This should not happen
        return [0] * 28 + [1, -1]
        
    return coeffs1 + coeffs2
