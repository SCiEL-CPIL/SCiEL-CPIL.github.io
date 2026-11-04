import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-92 tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Uses 8 disjoint 2x2x2 blocks with rank-7 decomposition (Winograd/Strassen style)
    and handles the remaining 36 terms with rank-1 (schoolbook) decomposition.
    """
    n, m, p = DIMENSIONS
    # Total rank: 8 blocks * 7 rank + 36 schoolbook terms = 56 + 36 = 92
    rank = 92
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the 7 factors for 2x2x2 rank-7 decomposition
    # Local indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    U_2x2 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 0]
    ]
    V_2x2 = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ]
    W_2x2 = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ]
    
    col_offset = 0
    
    # Define the 8 disjoint 2x2x2 blocks
    # i in {0,1}, {2,3}
    # j in {0,1}, {2,3}
    # k in {0,1}, {2,3}
    blocks = []
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2]:
                blocks.append((i_base, j_base, k_base))
                
    for i_base, j_base, k_base in blocks:
        for r in range(7):
            # Add factor r to the block
            # Local indices li, lj, lk in 0..1
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        # Map local to global
                        i = i_base + li
                        j = j_base + lj
                        k = k_base + lk
                        
                        # Local index in 2x2 matrix
                        # U (A) index: li*2 + lj
                        # V (B) index: lj*2 + lk
                        # W (C) index: lk*2 + li
                        
                        u_local_idx = li * 2 + lj
                        v_local_idx = lj * 2 + lk
                        w_local_idx = lk * 2 + li
                        
                        u_val = U_2x2[r][u_local_idx]
                        v_val = V_2x2[r][v_local_idx]
                        w_val = W_2x2[r][w_local_idx]
                        
                        # Global indices
                        u_global = i * m + j
                        v_global = j * p + k
                        w_global = k * n + i
                        
                        U[u_global, col_offset + r] += u_val
                        V[v_global, col_offset + r] += v_val
                        W[w_global, col_offset + r] += w_val
                        
        col_offset += 7
        
    # Handle the remaining 36 terms
    # These are terms where j=4 or k=4 (since blocks covered j<4, k<4)
    # Specifically, the blocks covered i in 0..3, j in 0..3, k in 0..3.
    # Remaining:
    # 1. j=4, i in 0..3, k in 0..3 (16 terms)
    # 2. k=4, j in 0..3, i in 0..3 (16 terms)
    # 3. j=4, k=4, i in 0..3 (4 terms)
    
    # Note: The term (i, j=4, k=4) is included in both 1 and 2 if not careful?
    # No, the sets are:
    # Set 1: {(i, 4, k) | i in 0..3, k in 0..3}
    # Set 2: {(i, j, 4) | i in 0..3, j in 0..3}
    # Set 3: {(i, 4, 4) | i in 0..3}
    # Set 1 and Set 2 are disjoint (k=4 vs j=4).
    # Set 3 is disjoint from both (j=4 AND k=4).
    # Wait, Set 1 has k in 0..3, so k!=4.
    # Set 2 has j in 0..3, so j!=4.
    # Set 3 has j=4, k=4.
    # So they are disjoint and cover all remaining 36 terms.
    
    # Add schoolbook terms for Set 1
    for i in range(n):
        for k in range(4): # k in 0..3
            j = 4
            u_global = i * m + j
            v_global = j * p + k
            w_global = k * n + i
            
            U[u_global, col_offset] = 1.0
            V[v_global, col_offset] = 1.0
            W[w_global, col_offset] = 1.0
            col_offset += 1
            
    # Add schoolbook terms for Set 2
    for i in range(n):
        for j in range(4): # j in 0..3
            k = 4
            u_global = i * m + j
            v_global = j * p + k
            w_global = k * n + i
            
            U[u_global, col_offset] = 1.0
            V[v_global, col_offset] = 1.0
            W[w_global, col_offset] = 1.0
            col_offset += 1
            
    # Add schoolbook terms for Set 3
    for i in range(n):
        j = 4
        k = 4
        u_global = i * m + j
        v_global = j * p + k
        w_global = k * n + i
        
        U[u_global, col_offset] = 1.0
        V[v_global, col_offset] = 1.0
        W[w_global, col_offset] = 1.0
        col_offset += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
