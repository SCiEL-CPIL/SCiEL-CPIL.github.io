import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 28 decomposition using Strassen's algorithm on 4 blocks of size 2x2
    # Each block requires 7 multiplications.
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    
    # Initialize factor matrices with zeros
    # U: (n*m, rank) = (4, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (16, 28)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for U (A indices: 00, 01, 10, 11)
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    
    U_terms = [
        [(0, 1), (3, 1)],       # M1
        [(2, 1), (3, 1)],       # M2
        [(0, 1)],               # M3
        [(3, 1)],               # M4
        [(0, 1), (1, 1)],       # M5
        [(2, 1), (0, -1)],      # M6
        [(1, 1), (3, -1)]       # M7
    ]
    
    # Strassen coefficients for V (B indices, local to block)
    # Local B indices: 00, 01, 10, 11
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    V_terms = [
        [(0, 1), (3, 1)],       # M1
        [(0, 1)],               # M2
        [(1, 1), (3, -1)],      # M3
        [(2, 1), (0, -1)],      # M4
        [(3, 1)],               # M5
        [(0, 1), (1, 1)],       # M6
        [(2, 1), (3, 1)]        # M7
    ]
    
    # Strassen coefficients for W (C indices, local to block)
    # Local C indices: 00, 10, 01, 11 (Note: C is p x n, so index is k*n + i)
    # M1: C00 + C11
    # M2: C10 - C11
    # M3: C01 + C11
    # M4: C00 + C10
    # M5: -C00 + C01
    # M6: C11
    # M7: C00
    
    W_terms = [
        [(0, 1), (3, 1)],       # M1
        [(1, 1), (3, -1)],      # M2
        [(2, 1), (3, 1)],       # M3
        [(0, 1), (1, 1)],       # M4
        [(0, -1), (2, 1)],      # M5
        [(3, 1)],               # M6
        [(0, 1)]                # M7
    ]
    
    # Iterate over the 4 blocks
    for block_idx in range(4):
        # Column offset in B and C
        col_offset = 2 * block_idx
        
        for s in range(7):
            r = block_idx * 7 + s
            
            # Fill U
            for idx, val in U_terms[s]:
                U[idx, r] += val
                
            # Fill V
            # Map local B index to global V index
            # Local index 0 -> (0, 0), 1 -> (0, 1), 2 -> (1, 0), 3 -> (1, 1)
            # Global index = j * p + (col_offset + k_local)
            for local_idx, val in V_terms[s]:
                j = local_idx // 2
                k_local = local_idx % 2
                global_idx = j * p + (col_offset + k_local)
                V[global_idx, r] += val
                
            # Fill W
            # Map local C index to global W index
            # Local index 0 -> (0, 0), 1 -> (1, 0), 2 -> (0, 1), 3 -> (1, 1)
            # Wait, previous code had:
            # Local C index 0 -> i=0, k=2*q (00)
            # Local C index 1 -> i=1, k=2*q (10)
            # Local C index 2 -> i=0, k=2*q+1 (01)
            # Local C index 3 -> i=1, k=2*q+1 (11)
            # W index = k * n + i
            for local_idx, val in W_terms[s]:
                # Based on previous mapping:
                # 0: i=0, k_local=0
                # 1: i=1, k_local=0
                # 2: i=0, k_local=1
                # 3: i=1, k_local=1
                i = local_idx % 2
                k_local = local_idx // 2
                k_global = col_offset + k_local
                global_idx = k_global * n + i
                W[global_idx, r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
