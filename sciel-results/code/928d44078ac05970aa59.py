import time
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s budget to be safe within 90s

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
            
    def lcm(a, b):
        return a // gcd(a, b) * b
        
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are integers
    weights = [L // d for d in denominators]
    
    # Split into 3 sets to keep list sizes manageable
    # A: 0..14 (15 items)
    # B: 15..30 (16 items)
    # C: 31..46 (16 items)
    
    A_weights = weights[0:15]
    B_weights = weights[15:31]
    C_weights = weights[31:47]
    
    # Generate subset sums for C
    # List of (sum, mask)
    sums_C = [(0, 0)]
    for i, w in enumerate(C_weights):
        current_len = len(sums_C)
        for k in range(current_len):
            s, m = sums_C[k]
            new_s = s + w
            # Pruning: if new_s >= L, we can skip? 
            # In the final combination, we need sum < L.
            # If a partial sum >= L, it can never be part of a valid solution 
            # because all weights are positive.
            if new_s < L:
                sums_C.append((new_s, m | (1 << i)))
                
    # Sort C by sum
    sums_C.sort(key=lambda x: x[0])
    
    # Generate subset sums for B and combine with C
    sums_B = [(0, 0)]
    for i, w in enumerate(B_weights):
        current_len = len(sums_B)
        for k in range(current_len):
            s, m = sums_B[k]
            new_s = s + w
            if new_s < L:
                sums_B.append((new_s, m | (1 << i)))
                
    # For each sum in B, find best sum in C
    # We want to maximize s_b + s_c < L
    # So for a given s_b, we want max s_c < L - s_b
    
    # Extract sorted sums from C for binary search
    C_sums = [s for s, m in sums_C]
    C_masks = [m for s, m in sums_C]
    
    sums_BC = []
    for s_b, m_b in sums_B:
        target = L - s_b
        # Find largest sum in C < target
        # bisect_left returns the index where target would be inserted
        # We want the element before that, if it exists and is < target
        idx = bisect.bisect_left(C_sums, target)
        if idx > 0:
            s_c = C_sums[idx - 1]
            m_c = C_masks[idx - 1]
            total_bc = s_b + s_c
            # total_bc is guaranteed < L because s_c < L - s_b
            sums_BC.append((total_bc, m_b | (m_c << len(B_weights))))
            
    # Sort BC by sum
    sums_BC.sort(key=lambda x: x[0])
    
    # Generate subset sums for A
    sums_A = [(0, 0)]
    for i, w in enumerate(A_weights):
        current_len = len(sums_A)
        for k in range(current_len):
            s, m = sums_A[k]
            new_s = s + w
            if new_s < L:
                sums_A.append((new_s, m | (1 << i)))
                
    # Extract sorted sums from BC for binary search
    BC_sums = [s for s, m in sums_BC]
    BC_masks = [m for s, m in sums_BC]
    
    best_total_sum = 0
    best_mask_a = 0
    best_mask_bc = 0
    
    for s_a, m_a in sums_A:
        target = L - s_a
        # Find largest sum in BC < target
        idx = bisect.bisect_left(BC_sums, target)
        if idx > 0:
            s_bc = BC_sums[idx - 1]
            m_bc = BC_masks[idx - 1]
            total_sum = s_a + s_bc
            
            if total_sum > best_total_sum:
                best_total_sum = total_sum
                best_mask_a = m_a
                best_mask_bc = m_bc
                
    # Reconstruct indices
    indices = []
    
    # From A
    for i in range(len(A_weights)):
        if best_mask_a & (1 << i):
            indices.append(i)
            
    # From B (stored in lower bits of best_mask_bc)
    for i in range(len(B_weights)):
        if best_mask_bc & (1 << i):
            indices.append(15 + i)
            
    # From C (stored in upper bits of best_mask_bc, shifted by len(B_weights))
    # Note: in sums_BC construction, we did m_b | (m_c << len(B_weights))
    # So m_c is in bits len(B_weights) to len(B_weights)+len(C_weights)-1
    for i in range(len(C_weights)):
        if best_mask_bc & (1 << (len(B_weights) + i)):
            indices.append(31 + i)
            
    # Verify and return
    if not indices:
        # Fallback
        return [0, 1, 5]
        
    # Exact check
    exact_sum = sum(Fraction(1, denominators[k]) for k in indices)
    if exact_sum >= 1:
        # This shouldn't happen if logic is correct, but fallback
        return [0, 1, 5]
        
    return indices
