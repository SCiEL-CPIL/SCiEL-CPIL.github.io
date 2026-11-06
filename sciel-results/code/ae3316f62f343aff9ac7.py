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
        Uses iterative doubling with preallocated arrays to reduce overhead.
        """
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            cur_len = len(sums)
            new_len = cur_len * 3
            new_sums = np.empty(new_len, dtype=np.int64)
            new_sums[0:cur_len] = sums - w
            new_sums[cur_len:2*cur_len] = sums
            new_sums[2*cur_len:3*cur_len] = sums + w
            sums = new_sums
        return sums

    # Generate sums for both halves
    # 3^15 is ~14.3 million
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums2 for binary search
    sums2_sorted = np.sort(sums2)
    
    len_s2 = len(sums2_sorted)
    
    # We want to find pairs (s1, s2) such that |s1 + s2| is minimized and non-zero.
    # Vectorized search with wider neighbor window for better coverage
    targets = -sums1
    
    # Find insertion points
    indices = np.searchsorted(sums2_sorted, targets)
    
    # Check neighbors: indices-2, indices-1, indices, indices+1, indices+2
    # This gives 5 candidates per sums1 entry
    num_candidates = 5
    offsets = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    
    # Create candidate index array of shape (num_candidates, len_sums1)
    N = len(sums1)
    candidate_indices = np.empty((num_candidates, N), dtype=np.int64)
    for k, off in enumerate(offsets):
        candidate_indices[k] = np.clip(indices + off, 0, len_s2 - 1)
    
    # Fetch corresponding sums2 values
    # candidate_indices shape: (num_candidates, N)
    # sums2_sorted[candidate_indices] gives shape (num_candidates, N)
    s2_candidates = sums2_sorted[candidate_indices]  # shape (num_candidates, N)
    
    # Compute totals: sums1 expanded + s2_candidates
    # sums1 has shape (N,), broadcast against (num_candidates, N)
    all_totals = s2_candidates + sums1[np.newaxis, :]  # shape (num_candidates, N)
    
    # Mask zeros and find minimum absolute non-zero value
    abs_totals = np.abs(all_totals)
    # Set zero entries to a large value
    abs_totals[all_totals == 0] = np.iinfo(np.int64).max
    
    # Find global minimum
    flat_idx = np.argmin(abs_totals)
    
    # Recover which candidate and which sums1 index
    row = flat_idx // N
    col = flat_idx % N
    
    s1_val = sums1[col]
    s2_val = s2_candidates[row, col]
        
    # Reconstruct coefficients using a faster approach
    # Store the coefficient encoding alongside sums during generation
    # For 15 elements, we can encode coefficients as a base-3 integer
    
    def generate_sums_with_coeffs(weights_half):
        """
        Generate all possible sums and their coefficient encodings.
        Returns (sums_array, coeffs_array) where coeffs_array encodes 
        coefficients as base-3 integers.
        """
        sums = np.array([0], dtype=np.int64)
        coeffs_enc = np.array([0], dtype=np.int64)
        
        for i, w in enumerate(weights_half):
            cur_len = len(sums)
            new_len = cur_len * 3
            new_sums = np.empty(new_len, dtype=np.int64)
            new_coeffs = np.empty(new_len, dtype=np.int64)
            
            # -w: coefficient = 0 at position i
            new_sums[0:cur_len] = sums - w
            new_coeffs[0:cur_len] = coeffs_enc  # add 0 * 3^i
            
            # 0: coefficient = 0 at position i
            new_sums[cur_len:2*cur_len] = sums
            new_coeffs[cur_len:2*cur_len] = coeffs_enc  # add 0 * 3^i
            
            # +w: coefficient = 1 at position i
            new_sums[2*cur_len:3*cur_len] = sums + w
            new_coeffs[2*cur_len:3*cur_len] = coeffs_enc + (3 ** i)
            
            sums = new_sums
            coeffs_enc = new_coeffs
        
        return sums, coeffs_enc
    
    def decode_coeffs(enc, n_half):
        """Decode base-3 encoding to list of coefficients in {-1, 0, 1}."""
        coeffs = []
        for i in range(n_half):
            digit = (enc // (3 ** i)) % 3
            if digit == 0:
                coeffs.append(0)
            elif digit == 1:
                coeffs.append(1)
            else:  # digit == 2, which represents -1
                coeffs.append(-1)
        return coeffs
    
    # Regenerate with coefficient tracking
    if time.time() < deadline - 10:
        sums1_c, enc1 = generate_sums_with_coeffs(w1)
        sums2_c, enc2 = generate_sums_with_coeffs(w2)
        
        # Sort sums2_c and keep track of encodings
        sort_idx = np.argsort(sums2_c)
        sums2_sorted_c = sums2_c[sort_idx]
        enc2_sorted = enc2[sort_idx]
        
        # Find best pair again with the sorted data
        targets = -sums1_c
        indices = np.searchsorted(sums2_sorted_c, targets)
        
        N = len(sums1_c)
        offsets = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
        candidate_indices = np.empty((5, N), dtype=np.int64)
        for k, off in enumerate(offsets):
            candidate_indices[k] = np.clip(indices + off, 0, len(sums2_sorted_c) - 1)
        
        s2_candidates = sums2_sorted_c[candidate_indices]
        enc2_candidates = enc2_sorted[candidate_indices]
        
        all_totals = s2_candidates + sums1_c[np.newaxis, :]
        abs_totals = np.abs(all_totals)
        abs_totals[all_totals == 0] = np.iinfo(np.int64).max
        
        flat_idx = np.argmin(abs_totals)
        row = flat_idx // N
        col = flat_idx % N
        
        s1_val = sums1_c[col]
        enc1_val = enc1[col]
        enc2_val = enc2_candidates[row, col]
        
        coeffs1 = decode_coeffs(enc1_val, len(w1))
        coeffs2 = decode_coeffs(enc2_val, len(w2))
        
        if coeffs1 is None or coeffs2 is None:
            return [0] * 28 + [1, -1]
        
        return coeffs1 + coeffs2
    else:
        # Fallback: use the original approach with dict-based reconstruction
        def find_coeffs(weights_half, target_sum):
            n_half = len(weights_half)
            mid_h = n_half // 2
            w_a = weights_half[:mid_h]
            w_b = weights_half[mid_h:]
            
            sums_a = {0: []}
            for i, w in enumerate(w_a):
                new_sums = {}
                for s, coeffs in sums_a.items():
                    for c in [-1, 0, 1]:
                        ns = s + c * w
                        new_coeffs = coeffs + [c]
                        if ns not in new_sums:
                            new_sums[ns] = new_coeffs
                sums_a = new_sums
                
            sums_b = {0: []}
            for i, w in enumerate(w_b):
                new_sums = {}
                for s, coeffs in sums_b.items():
                    for c in [-1, 0, 1]:
                        ns = s + c * w
                        new_coeffs = coeffs + [c]
                        if ns not in new_sums:
                            new_sums[ns] = new_coeffs
                sums_b = new_sums
                
            for s_b, coeffs_b in sums_b.items():
                target_a = target_sum - s_b
                if target_a in sums_a:
                    return sums_a[target_a] + coeffs_b
            
            return None

        coeffs1 = find_coeffs(w1, s1_val)
        coeffs2 = find_coeffs(w2, s2_val)
        
        if coeffs1 is None or coeffs2 is None:
            return [0] * 28 + [1, -1]
        
        return coeffs1 + coeffs2
