import numpy as np
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # LCM of 1 to 30
    # 2329089562800
    L = 2329089562800
    
    # We want to find coefficients c_i in {-1, 0, 1} for i=1..30
    # such that S = sum(c_i / i) is non-zero and |S| is minimized.
    # Let N = L * S = sum(c_i * L / i).
    # We want to find the smallest non-zero |N| achievable.
    
    # DP: Set of achievable numerators.
    # To save memory/time, we can store the set of reachable values.
    # We also need to recover the coefficients.
    # Storing parents for ~10^6 to 10^7 states might be heavy but feasible within 60s if optimized.
    # However, just finding the min value first is faster.
    # Then we can backtrack if needed, or store parents from the start.
    
    # Let's try to store the set of reachable numerators first to find the target.
    # If the set is too large, we might need a different strategy.
    # Expected size: The number of distinct sums is bounded by the range / gcd.
    # Range is approx 9e12. But sums are sparse.
    
    # Optimization: We can split into two halves again but with exact integers?
    # Meet-in-the-middle with exact integers:
    # Half 1: 15 terms. 3^15 = 14,348,907 sums.
    # Half 2: 15 terms. 3^15 = 14,348,907 sums.
    # We can generate all sums for half 1, sort them.
    # Generate all sums for half 2, sort them.
    # We want sum1 + sum2 = N_min_nonzero.
    # Actually, we want to find min |s1 + s2| != 0.
    # This is the classic "closest pair" problem on two sorted lists, but we need to exclude 0.
    
    # This is much faster than full DP over 30 terms (which would be 3^30) and avoids floating point issues.
    # 14M integers is manageable in memory and time.
    
    def gen_exact_sums(n_terms, start_idx):
        sums = [0]
        coeffs = [tuple([0]*n_terms)] # Store coeffs to retrieve solution later? 
        # Storing coeffs for 14M items might be memory heavy (14M * 15 bytes ~ 200MB).
        # Let's just store sums first to find the best N.
        # Then we can re-run or store parents to recover coeffs.
        # Better: Store (sum, coeffs) or use a dictionary?
        # Actually, we can just store the sums, find the best combination, 
        # and then do a second pass to recover the coefficients for that specific sum?
        # Or store the path?
        
        # Let's store the sums and the coefficients as a compact representation or just recompute.
        # Recomputing is safer for memory.
        # But we need to know WHICH coefficients produce the sum.
        
        # Alternative: Store the sum and the index of the parent in the previous step?
        # This is a standard DP tree.
        
        # Let's try generating the lists of (sum, coeffs_tuple) for 15 terms.
        # 14M tuples of 15 ints is ~ 14M * 15 * 8 bytes (if int32) = 1.7GB. Too much.
        
        # We can store the sum and a bitstring or small int encoding the coeffs?
        # Coeffs are -1, 0, 1. 15 coeffs can be encoded in 2 bits each = 30 bits. Fits in int64.
        
        sums = [0]
        encodings = [0] # Bit encoding of coeffs
        
        for i in range(n_terms):
            term_num = L // (start_idx + i + 1)
            new_sums = []
            new_encs = []
            
            # Optimization: Use lists and extend, or pre-allocate?
            # 3x growth.
            
            # We iterate through existing sums
            # For each, we add -term, 0, +term
            
            # To avoid overhead of Python loops for 14M items, we might need numpy.
            # But numpy doesn't handle variable length lists well during growth.
            # However, 15 steps is small.
            
            # Let's use numpy for the sums to speed up the addition, but we need to track encodings.
            # It might be faster to just use Python lists if we are careful, but 14M iterations in Python is slow (~1s per 10M ops? 14M * 3 ops = 42M ops. Might be 4-5 seconds).
            # Total time budget 50s. 15 terms * 2 halves = 30 steps.
            # Each step processes 3^(k-1) items.
            # Total items processed in one half: sum_{k=0}^{14} 3^k = (3^15 - 1) / 2 ~ 7M.
            # Two halves: 14M items processed in loops.
            # Python loop overhead is high.
            
            # Let's try to vectorize with numpy where possible.
            # But the encoding part is tricky to vectorize easily with variable length.
            
            # Actually, we can just store the sums in a numpy array and the encodings in a numpy array.
            
            pass # Placeholder, will implement in the edit below

    # Revised plan for gen_exact_sums:
    # Use numpy arrays for sums and encodings.
    
    def gen_exact_sums_vec(n_terms, start_idx):
        sums = np.array([0], dtype=np.int64)
        encs = np.array([0], dtype=np.int64)
        
        for i in range(n_terms):
            term_num = L // (start_idx + i + 1)
            curr_size = len(sums)
            
            # New sizes
            new_sums = np.empty(curr_size * 3, dtype=np.int64)
            new_encs = np.empty(curr_size * 3, dtype=np.int64)
            
            # 1. Coeff 0
            new_sums[:curr_size] = sums
            new_encs[:curr_size] = encs
            
            # 2. Coeff 1
            new_sums[curr_size:2*curr_size] = sums + term_num
            # Encoding: set bit i (2*i and 2*i+1) to 01? 
            # Let's define encoding: 00 for 0, 01 for 1, 10 for -1? 
            # Or simpler: store the value directly in a base-3 representation?
            # Base 3 representation of coefficients:
            # c_0 is least significant digit.
            # val = sum(c_j * 3^j).
            # c_j in {-1, 0, 1}.
            # We can map -1->0, 0->1, 1->2 for base 3 digits?
            # No, that changes the sum.
            # Let's just use 2 bits per coefficient.
            # Bit 2*i: 0 if c_i >= 0, 1 if c_i < 0?
            # Bit 2*i+1: 0 if c_i == 0, 1 if c_i != 0?
            # 00 -> 0
            # 01 -> 1
            # 10 -> -1
            # 11 -> invalid
            
            # For coeff 1:
            # Set bits 2*i and 2*i+1 to 01.
            # Mask for bit 2*i+1 is (1 << (2*i + 1)).
            # We want to set that bit.
            bit_1 = 1 << (2*i + 1)
            new_encs[curr_size:2*curr_size] = encs | bit_1
            
            # 3. Coeff -1
            new_sums[2*curr_size:] = sums - term_num
            # Set bits 2*i and 2*i+1 to 10.
            # Bit 2*i is 1, Bit 2*i+1 is 0.
            bit_neg = 1 << (2*i)
            new_encs[2*curr_size:] = encs | bit_neg
            
            sums = new_sums
            encs = new_encs
            
        return sums, encs

    # Generate for first half (1..15)
    sums_A, encs_A = gen_exact_sums_vec(15, 0)
    
    if time.time() > deadline:
         # Fallback
         return [0]*28 + [1, -1]

    # Generate for second half (16..30)
    sums_B, encs_B = gen_exact_sums_vec(15, 15)
    
    if time.time() > deadline:
         # Fallback
         return [0]*28 + [1, -1]

    # Sort sums_B
    idx_sort = np.argsort(sums_B)
    sums_B_sorted = sums_B[idx_sort]
    encs_B_sorted = encs_B[idx_sort]
    
    # We want to find min |s_A + s_B| != 0.
    # For each s_A, we want to find s_B such that s_A + s_B is close to 0.
    # Target s_B = -s_A.
    
    # Use searchsorted on sums_B_sorted for -sums_A
    targets = -sums_A
    idx_B = np.searchsorted(sums_B_sorted, targets)
    
    len_B = len(sums_B_sorted)
    
    # Check idx and idx-1
    # Need to handle boundaries
    
    # Vectorized check
    # We create candidates for each A.
    
    # Candidate 1: idx_B
    # Candidate 2: idx_B - 1
    
    # Clip indices
    idx_c1 = np.clip(idx_B, 0, len_B - 1)
    idx_c2 = np.clip(idx_B - 1, 0, len_B - 1)
    
    # Get corresponding sums and encs
    s_B_c1 = sums_B_sorted[idx_c1]
    enc_B_c1 = encs_B_sorted[idx_c1]
    
    s_B_c2 = sums_B_sorted[idx_c2]
    enc_B_c2 = encs_B_sorted[idx_c2]
    
    # Calculate total sums
    total_c1 = sums_A + s_B_c1
    total_c2 = sums_A + s_B_c2
    
    # Calculate absolute values
    abs_c1 = np.abs(total_c1)
    abs_c2 = np.abs(total_c2)
    
    # Mask zeros
    abs_c1[total_c1 == 0] = np.inf
    abs_c2[total_c2 == 0] = np.inf
    
    # Find global min
    # Combine into one array? Or find min of mins?
    min_c1 = np.min(abs_c1)
    min_c2 = np.min(abs_c2)
    
    if min_c1 < min_c2:
        min_val = min_c1
        min_idx_A = np.argmin(abs_c1)
        min_idx_B = idx_c1[min_idx_A]
        best_enc_A = encs_A[min_idx_A]
        best_enc_B = encs_B_sorted[min_idx_B]
    else:
        min_val = min_c2
        min_idx_A = np.argmin(abs_c2)
        min_idx_B = idx_c2[min_idx_A]
        best_enc_A = encs_A[min_idx_A]
        best_enc_B = encs_B_sorted[min_idx_B]
        
    if np.isinf(min_val):
        return [0]*28 + [1, -1]
        
    # Decode coefficients
    # best_enc_A corresponds to terms 1..15
    # best_enc_B corresponds to terms 16..30
    
    coeffs = [0] * 30
    
    # Decode A
    for i in range(15):
        # Extract 2 bits at position 2*i
        val = (best_enc_A >> (2*i)) & 3
        if val == 0:
            coeffs[i] = 0
        elif val == 1:
            coeffs[i] = 1
        elif val == 2:
            coeffs[i] = -1
        else:
            coeffs[i] = 0 # Should not happen
            
    # Decode B
    for i in range(15):
        # Extract 2 bits at position 2*i
        val = (best_enc_B >> (2*i)) & 3
        if val == 0:
            coeffs[15+i] = 0
        elif val == 1:
            coeffs[15+i] = 1
        elif val == 2:
            coeffs[15+i] = -1
        else:
            coeffs[15+i] = 0
            
    return coeffs
