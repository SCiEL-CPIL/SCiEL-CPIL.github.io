import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
        
    # Weights w[i] = L / (i+1) for i in 0..29
    # c[i] corresponds to 1/(i+1)
    w = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # Split into two halves
    half = n // 2
    w_A = w[:half]
    w_B = w[half:]
    
    def generate_sums(weights):
        """
        Generate all possible signed sums for the given weights.
        Returns a numpy array of int64.
        """
        sums = np.array([0], dtype=np.int64)
        for w_val in weights:
            # For each existing sum, we can subtract, keep, or add the weight
            # This creates 3 * len(sums) new entries
            sums = np.concatenate([sums - w_val, sums, sums + w_val])
        return sums

    # Generate sums for first half
    sums_A = generate_sums(w_A)
    # Generate sums for second half
    sums_B = generate_sums(w_B)
    
    # Sort sums_A for binary search
    sums_A.sort()
    
    best_abs_K = np.iinfo(np.int64).max
    best_coeffs = None
    
    # We need to reconstruct coefficients later, so we might need to store indices or re-search.
    # However, for now, let's just find the minimum value.
    # To reconstruct, we can do a second pass or store the indices of the best match.
    
    # Optimization: If we find 1, we can stop early.
    
    # Iterate through sums_B
    # To speed up, we can iterate in chunks or just use a simple loop.
    # 3^15 = 14,348,907 iterations.
    
    # We also need to track the coefficients that produced the best sum.
    # Reconstructing coefficients from the sum value alone is hard (inverse problem).
    # So we should store the index of the best sum in A and B, or the sums themselves.
    
    # Let's store the index of the best A and B sums.
    best_idx_A = -1
    best_idx_B = -1
    
    # To map index back to coefficients, we need a way to decode the index.
    # The generation order is:
    # sums = [0]
    # for w in weights:
    #   sums = [s-w for s in sums] + [s for s in sums] + [s+w for s in sums]
    # This is a base-3 representation essentially, but with a specific ordering.
    # Index 0: all 0
    # Index 1: last weight +1, rest 0? No.
    # Let's trace:
    # Start: [0]
    # w1: [-w1, 0, w1]
    # w2: [-w1-w2, -w1, -w1+w2, 0-w2, 0, 0+w2, w1-w2, w1, w1+w2]
    # The index in the final array corresponds to a base-3 number if we look at the choices.
    # Actually, it's easier to just regenerate or store the coefficients during generation?
    # Storing 14M * 15 coeffs is too much memory.
    
    # Alternative: Store the index.
    # The index in the generated array `sums` corresponds to a specific combination.
    # If we generate recursively or iteratively, the index $I$ in the final list of size $3^k$
    # can be decoded into $k$ digits in base 3.
    # Let's verify the mapping.
    # Step 0: [0] (size 1)
    # Step 1: [-w1, 0, w1] (size 3)
    #   Index 0 -> -w1 -> coeff -1
    #   Index 1 -> 0    -> coeff 0
    #   Index 2 -> w1   -> coeff 1
    # Step 2: 
    #   From -w1: [-w1-w2, -w1, -w1+w2] (Indices 0,1,2)
    #   From 0:   [-w2, 0, w2]          (Indices 3,4,5)
    #   From w1:  [w1-w2, w1, w1+w2]    (Indices 6,7,8)
    #   
    # So, index $I$ in the final array of size $3^k$:
    # Let $I = d_0 + 3 d_1 + \dots + 3^{k-1} d_{k-1}$?
    # No, the concatenation order is:
    # New List = [Old - w, Old, Old + w]
    # So the first 1/3 comes from "subtract", next 1/3 from "keep", last 1/3 from "add".
    # This applies to the LAST weight processed.
    # So if we process weights $w_1, \dots, w_k$:
    # The most significant digit in base 3 corresponds to $w_k$?
    # Let's check.
    # Final array is formed by taking the previous array (size $3^{k-1}$) and expanding.
    # The index $I$ in the final array of size $3^k$:
    # $I \pmod 3$ determines the choice for the LAST weight $w_k$?
    # No, the expansion is:
    # `sums_new = concat(sums_old - w, sums_old, sums_old + w)`
    # If $I < 3^{k-1}$, the choice for $w_k$ is -1, and the rest is determined by $I$ in `sums_old`.
    # If $3^{k-1} \le I < 2 \cdot 3^{k-1}$, the choice for $w_k$ is 0, and the rest is $I - 3^{k-1}$ in `sums_old`.
    # If $I \ge 2 \cdot 3^{k-1}$, the choice for $w_k$ is 1, and the rest is $I - 2 \cdot 3^{k-1}$ in `sums_old`.
    
    # So we can decode the index to get the coefficients.
    
    def decode_index(idx, k):
        """
        Decode an index in the generated sums array back to coefficients.
        k is the number of weights.
        Returns a list of coefficients of length k.
        The weights were processed in order w[0], w[1], ..., w[k-1].
        """
        coeffs = [0] * k
        size = 1
        # We process from the last weight (most significant in the concat structure)
        # Actually, the structure is recursive on the last step.
        # So we decode from the last weight to the first.
        for j in range(k - 1, -1, -1):
            # size is 3^j at this step? No.
            # At step j+1 (processing w[j]), the size of the block is 3^j.
            # Wait, the initial size is 1.
            # After w[0], size is 3.
            # After w[1], size is 9.
            # ...
            # After w[k-1], size is 3^k.
            
            # In the final array of size 3^k, the "chunks" are of size 3^(k-1).
            # So we divide by 3^(k-1) to get the digit for w[k-1].
            block_size = 3 ** j
            # The index within the current block determines the previous weights.
            # The quotient determines the choice for the current weight w[j].
            
            choice = idx // block_size
            idx = idx % block_size
            
            if choice == 0:
                coeffs[j] = -1
            elif choice == 1:
                coeffs[j] = 0
            else:
                coeffs[j] = 1
                
        return coeffs

    # Search for minimum |sum_A + sum_B|
    # We iterate through sums_B.
    # To avoid Python loop overhead, we can try to vectorize parts, but binary search is tricky.
    # Given 14M items, a Python loop is acceptable if it's tight.
    
    # We can speed it up by only checking a subset if we are confident, but let's do full search.
    
    # Optimization: Sort sums_B as well? No need.
    
    # Use np.searchsorted for each b.
    # To make it faster, we can process in chunks? No, searchsorted is fast enough.
    
    # Let's run the loop.
    for i_b, b_val in enumerate(sums_B):
        if time.time() > deadline - 5.0:
            break
            
        # Find position in sums_A where -b_val would be inserted
        target = -b_val
        idx = np.searchsorted(sums_A, target)
        
        # Check idx and idx-1
        # We need to ensure indices are within bounds
        
        if idx < len(sums_A):
            a_val = sums_A[idx]
            K = a_val + b_val
            if K != 0:
                abs_K = abs(K)
                if abs_K < best_abs_K:
                    best_abs_K = abs_K
                    best_idx_A = idx
                    best_idx_B = i_b
                    if best_abs_K == 1:
                        break
        
        if idx > 0:
            a_val = sums_A[idx - 1]
            K = a_val + b_val
            if K != 0:
                abs_K = abs(K)
                if abs_K < best_abs_K:
                    best_abs_K = abs_K
                    best_idx_A = idx - 1
                    best_idx_B = i_b
                    if best_abs_K == 1:
                        break

    # Reconstruct coefficients
    if best_idx_A != -1 and best_idx_B != -1:
        coeffs_A = decode_index(best_idx_A, half)
        coeffs_B = decode_index(best_idx_B, n - half)
        coeffs = coeffs_A + coeffs_B
    else:
        # Fallback: 1/30
        coeffs = [0] * (n - 1) + [1]
        
    return coeffs

# EVOLVE-BLOCK-END
