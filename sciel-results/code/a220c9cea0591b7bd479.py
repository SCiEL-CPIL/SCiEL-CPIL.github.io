import math
import time
import bisect
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns exactly 30 integer coefficients from {-1, 0, 1} such that
    sum(coeff[i-1]/i for i in 1..30) is nonzero and has minimal absolute value.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 30
    half1 = 15
    half2 = 15
    
    # Calculate LCM of 1 to 30
    def lcm(a, b):
        return a // math.gcd(a, b) * b
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Scaled weights: weight[i] = L // (i+1)
    # Position i corresponds to denominator i+1
    weights = [L // (i + 1) for i in range(n)]
    
    def enumerate_half_dict(start, length):
        """
        Generates a dictionary mapping sum -> coeffs tuple for positions start..start+length-1.
        Uses iterative expansion.
        """
        d = {0: ()}
        for pos in range(start, start + length):
            w = weights[pos]
            new_d = {}
            for s, c in d.items():
                # coeff = 0
                key0 = s
                if key0 not in new_d:
                    new_d[key0] = c + (0,)
                # coeff = 1
                key1 = s + w
                if key1 not in new_d:
                    new_d[key1] = c + (1,)
                # coeff = -1
                keym1 = s - w
                if keym1 not in new_d:
                    new_d[keym1] = c + (-1,)
            d = new_d
        return d

    # Enumerate first half
    d1 = enumerate_half_dict(0, half1)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
        
    # Enumerate second half
    d2 = enumerate_half_dict(half1, half2)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Convert d2 to NumPy arrays for vectorized search
    # d2 keys are integers, values are tuples of 15 ints
    keys2_list = list(d2.keys())
    vals2_list = list(d2.values())
    
    # Sort by keys
    sorted_indices = np.argsort(keys2_list)
    keys2_arr = np.array([keys2_list[i] for i in sorted_indices], dtype=np.int64)
    vals2_arr = np.array([vals2_list[i] for i in sorted_indices], dtype=np.int8)
    
    # Prepare d1 keys and values
    keys1_list = list(d1.keys())
    vals1_list = list(d1.values())
    keys1_arr = np.array(keys1_list, dtype=np.int64)
    vals1_arr = np.array(vals1_list, dtype=np.int8)
    
    # Vectorized search
    # For each key in d1, find the position in d2 keys that sums to ~0
    # target = -key1
    targets = -keys1_arr
    idx = np.searchsorted(keys2_arr, targets)
    
    # We need to check idx-1, idx, idx+1
    # Handle boundaries
    idx_minus = np.clip(idx - 1, 0, len(keys2_arr) - 1)
    idx_self = np.clip(idx, 0, len(keys2_arr) - 1)
    idx_plus = np.clip(idx + 1, 0, len(keys2_arr) - 1)
    
    # Create array of candidate indices
    # Shape: (3, N)
    cand_indices = np.vstack([idx_minus, idx_self, idx_plus])
    
    # Get corresponding keys from d2
    # keys2_arr[cand_indices] will have shape (3, N)
    keys2_cand = keys2_arr[cand_indices]
    
    # Calculate sums
    # keys1_arr is (N,), keys2_cand is (3, N)
    # We need to broadcast keys1_arr to (3, N)
    sums = keys1_arr[None, :] + keys2_cand
    
    # Calculate absolute values
    abs_sums = np.abs(sums)
    
    # Exclude zero sums
    abs_sums[sums == 0] = np.inf
    
    # Find the minimum
    min_abs = np.min(abs_sums)
    min_idx_flat = np.argmin(abs_sums)
    min_row, min_col = np.unravel_index(min_idx_flat, abs_sums.shape)
    
    # Retrieve coefficients
    best_key1 = keys1_arr[min_col]
    best_key2_idx = cand_indices[min_row, min_col]
    best_key2 = keys2_arr[best_key2_idx]
    
    best_val1 = vals1_arr[min_col]
    best_val2 = vals2_arr[best_key2_idx]
    
    coeffs = np.concatenate([best_val1, best_val2]).tolist()
    
    # Final verification
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(coeffs))
    if total_frac == 0:
        # Fallback
        return [0] * 28 + [1, -1]
        
    return coeffs

# EVOLVE-BLOCK-END
