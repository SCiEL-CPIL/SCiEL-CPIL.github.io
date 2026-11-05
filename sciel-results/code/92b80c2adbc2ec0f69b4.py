import math
import numpy as np

def construct():
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)
        
    # Weights: L // d
    W = [L // d for d in denominators]
    
    # Split into two halves
    # 24 and 23
    n1 = 24
    n2 = n - n1
    
    W1 = W[:n1]
    W2 = W[n1:]
    
    # --- Generate subset sums for W2 (smaller half) ---
    size2 = 1 << n2
    S2 = np.zeros(size2, dtype=np.int64)
    M2 = np.arange(size2, dtype=np.int64)
    
    # Generate sums for W2
    # Using iterative bit trick
    for m in range(1, size2):
        lbit = m & -m
        idx = lbit.bit_length() - 1
        S2[m] = S2[m ^ lbit] + W2[idx]
        
    # Sort S2 and keep M2 aligned
    idx_sort = np.argsort(S2)
    S2_sorted = S2[idx_sort]
    M2_sorted = M2[idx_sort]
    
    # --- Generate subset sums for W1 (larger half) ---
    size1 = 1 << n1
    S1 = np.zeros(size1, dtype=np.int64)
    M1 = np.arange(size1, dtype=np.int64)
    
    for m in range(1, size1):
        lbit = m & -m
        idx = lbit.bit_length() - 1
        S1[m] = S1[m ^ lbit] + W1[idx]
        
    # Filter S1 to only include sums < L
    # This is crucial for performance and correctness (avoiding negative targets)
    valid_mask = S1 < L
    S1_valid = S1[valid_mask]
    M1_valid = M1[valid_mask]
    
    if len(S1_valid) == 0:
        # Fallback: empty set
        return []
        
    # Target for S2: T = L - S1
    # We want max S2 < T
    T = L - S1_valid
    
    # Find position in S2_sorted
    # searchsorted returns index where T would be inserted to maintain order
    # side='right' gives index of first element >= T
    # We want largest element < T, so index - 1
    pos = np.searchsorted(S2_sorted, T, side='right') - 1
    
    # Filter out invalid positions (pos < 0)
    valid_pos_mask = pos >= 0
    if not np.any(valid_pos_mask):
        return []
        
    S1_cand = S1_valid[valid_pos_mask]
    M1_cand = M1_valid[valid_pos_mask]
    pos_cand = pos[valid_pos_mask]
    
    # Get corresponding S2 values
    S2_cand = S2_sorted[pos_cand]
    M2_cand = M2_sorted[pos_cand]
    
    # Total sums
    Total = S1_cand + S2_cand
    
    # Ensure strictly below L (should be guaranteed by S2 < T = L - S1, but check anyway)
    # Actually, if S2 < L - S1, then S1 + S2 < L.
    # Since S2_sorted[pos] < T, this holds.
    
    # Find the maximum total sum
    max_idx = np.argmax(Total)
    
    best_total = Total[max_idx]
    best_m1 = M1_cand[max_idx]
    best_m2 = M2_cand[max_idx]
    
    # Reconstruct indices
    selected_indices = []
    
    # Indices from W1 (0 to n1-1)
    m = best_m1
    for i in range(n1):
        if m & (1 << i):
            selected_indices.append(i)
            
    # Indices from W2 (n1 to n-1)
    m = best_m2
    for i in range(n2):
        if m & (1 << i):
            selected_indices.append(n1 + i)
            
    # The problem asks for zero-based indices into denominators [2,3,...,48]
    # Our indices 0..46 correspond to denominators 2..48.
    # So we return the list of indices.
    
    # Sanity check: sum must be < 1
    # We used integer arithmetic with L, so best_total < L is guaranteed.
    
    return selected_indices

# EVOLVE-BLOCK-END
