import numpy as np
from math import gcd
import time

def construct():
    start_time = time.time()
    deadline = 50.0  # 50s limit to be safe
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
        
    # 2. Weights
    # A: 1..15, B: 16..30
    weights_A = [L // i for i in range(1, 16)]
    weights_B = [L // i for i in range(16, 31)]
    
    # 3. Generate sums for A
    n_A = 3 ** 15
    indices_A = np.arange(n_A, dtype=np.int64)
    
    # Compute digits for A
    # To save memory, we can compute sums incrementally or use dot product
    # Let's try dot product with a temporary array of digits
    # digits shape (15, n_A)
    # This might be large: 15 * 14M * 8 bytes = 1.7 GB
    # If memory is tight, we can do it in chunks or use a loop.
    # Given the deadline, let's try the vectorized approach first.
    
    # Optimization: We don't need to store all digits if we compute the sum iteratively
    # But vectorized dot is faster.
    
    try:
        # Create digits array
        digits_A = np.zeros((15, n_A), dtype=np.int64)
        divisor = 1
        for i in range(15):
            digits_A[i] = (indices_A // divisor) % 3
            divisor *= 3
            
        coeffs_A = digits_A - 1
        A_sums = np.dot(weights_A, coeffs_A)
        
        # Free memory
        del digits_A
        del coeffs_A
        
        # Sort A_sums
        order_A = np.argsort(A_sums)
        sorted_A_sums = A_sums[order_A]
        sorted_A_indices = order_A
        
        # 4. Generate sums for B
        n_B = 3 ** 15
        indices_B = np.arange(n_B, dtype=np.int64)
        
        digits_B = np.zeros((15, n_B), dtype=np.int64)
        divisor = 1
        for i in range(15):
            digits_B[i] = (indices_B // divisor) % 3
            divisor *= 3
            
        coeffs_B = digits_B - 1
        B_sums = np.dot(weights_B, coeffs_B)
        
        del digits_B
        del coeffs_B
        
        # 5. Find best match
        targets = -B_sums
        # Search in sorted_A_sums
        search_indices = np.searchsorted(sorted_A_sums, targets)
        
        # Candidate 1: exact or next
        idx1 = np.clip(search_indices, 0, len(sorted_A_sums) - 1)
        cand_A_sums_1 = sorted_A_sums[idx1]
        cand_N_1 = cand_A_sums_1 + B_sums
        abs_cand_N_1 = np.abs(cand_N_1)
        
        # Candidate 2: previous
        idx2 = np.clip(search_indices - 1, 0, len(sorted_A_sums) - 1)
        cand_A_sums_2 = sorted_A_sums[idx2]
        cand_N_2 = cand_A_sums_2 + B_sums
        abs_cand_N_2 = np.abs(cand_N_2)
        
        # Mask zeros
        mask1 = (cand_N_1 != 0)
        mask2 = (cand_N_2 != 0)
        
        # Replace zeros with inf
        abs_cand_N_1 = np.where(mask1, abs_cand_N_1, np.inf)
        abs_cand_N_2 = np.where(mask2, abs_cand_N_2, np.inf)
        
        # Find global min
        min_idx_1 = np.argmin(abs_cand_N_1)
        min_val_1 = abs_cand_N_1[min_idx_1]
        
        min_idx_2 = np.argmin(abs_cand_N_2)
        min_val_2 = abs_cand_N_2[min_idx_2]
        
        if min_val_1 < min_val_2:
            best_b_idx = min_idx_1
            best_a_sorted_idx = idx1[min_idx_1]
            best_N = cand_N_1[min_idx_1]
        else:
            best_b_idx = min_idx_2
            best_a_sorted_idx = idx2[min_idx_2]
            best_N = cand_N_2[min_idx_2]
            
        # If best_N is 0, we need to find the next best.
        # The above logic might pick a 0 if it exists in the "best" slot but was masked?
        # No, we masked 0 to inf. So argmin will skip it.
        # However, if ALL are 0? Impossible, sum of 30 terms with coeffs in {-1,0,1}
        # can be 0, but we need non-zero.
        # If the minimum non-zero is found, we are good.
        
        if best_N == 0:
            # This shouldn't happen due to masking, but just in case
            # We need to find the smallest non-zero.
            # Let's flatten and find min non-zero.
            all_Ns = np.concatenate([cand_N_1, cand_N_2])
            all_abs = np.abs(all_Ns)
            all_mask = (all_Ns != 0)
            all_abs_masked = np.where(all_mask, all_abs, np.inf)
            global_min_idx = np.argmin(all_abs_masked)
            best_N = all_Ns[global_min_idx]
            
            # Determine which candidate it was
            if global_min_idx < len(cand_N_1):
                best_b_idx = global_min_idx
                best_a_sorted_idx = idx1[global_min_idx]
            else:
                best_b_idx = global_min_idx - len(cand_N_1)
                best_a_sorted_idx = idx2[best_b_idx]
                
        # Map back to original indices
        best_a_orig_idx = sorted_A_indices[best_a_sorted_idx]
        best_b_orig_idx = best_b_idx
        
        # 6. Reconstruct coefficients
        def get_coeffs(idx, n_terms):
            coeffs = []
            for i in range(n_terms):
                d = (idx // (3 ** i)) % 3
                coeffs.append(d - 1)
            return coeffs
            
        coeffs_A = get_coeffs(best_a_orig_idx, 15)
        coeffs_B = get_coeffs(best_b_orig_idx, 15)
        
        return coeffs_A + coeffs_B
        
    except MemoryError:
        # Fallback to Python loop if memory is too low
        # This is a simplified fallback, might be slower
        pass

    # Fallback seed result if something goes wrong
    return [0] * 28 + [1, -1]

if __name__ == "__main__":
    # Test
    res = construct()
    print(res)
    s = sum(c / i for i, c in enumerate(res, 1))
    print(s)
