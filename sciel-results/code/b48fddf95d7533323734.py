"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # The multiplication tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    
    # Strategy: Use block structure with 2x2 blocks
    # A is 4x4 -> 2x2 blocks (A00, A01, A10, A11) each 2x2
    # B is 4x6 -> 2x3 blocks (B00,B01,B02, B10,B11,B12) each 2x2  
    # C is 4x6 -> 2x3 blocks (C00,C01,C02, C10,C11,C12) each 2x2
    
    # C[i1,k1] = sum_{j1=0,1} A[i1,j1]*B[j1,k1]
    # In block form: C[i1,k1] = A[i1,0]*B[0,k1] + A[i1,1]*B[1,k1]
    
    # For each (i1,k1) pair, we have a sum of two 2x2 matrix products.
    # Let's use a direct construction with shared Strassen-like terms.
    
    # Actually, let me just build a known good decomposition.
    # For 4x4x6, a reasonable upper bound is around 48-64.
    
    # Let me use the following approach:
    # For each output position (i,k) where i in 0..3, k in 0..5:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # We can group these into bilinear forms.
    
    # Simple approach: rank 48 using block structure
    # Each C block (2x2) = A_block1 * B_block1 + A_block2 * B_block2
    # Each 2x2 product = 4 rank-1 tensors (schoolbook)
    # So each C block = 8 rank-1 tensors
    # 6 C blocks * 8 = 48
    
    # But we can potentially do better by sharing terms.
    
    # Let me implement a rank-48 decomposition first, then try to reduce.
    
    rank = 48
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # For each C block (i1, k1) where i1 in {0,1}, k1 in {0,1,2}:
    for i1 in range(2):
        for k1 in range(3):
            # C[i1*2+0, k1*2+0] and C[i1*2+1, k1*2+0]
            # C[i1*2+0, k1*2+1] and C[i1*2+1, k1*2+1]
            
            # This C block depends on A rows i1*2, i1*2+1 and B cols k1*2, k1*2+1
            # And on B rows 0,1,2,3
            
            # C[i1*2+i2, k1*2+k2] = sum_{j1=0,1} A[i1*2+i2, j1*2+j2] * B[j1*2+j2, k1*2+k2]
            
            # For j1=0: A[i1*2+i2, j2] * B[j2, k1*2+k2] for j2 in {0,1}
            # For j1=1: A[i1*2+i2, 2+j2] * B[2+j2, k1*2+k2] for j2 in {0,1}
            
            # So 4 terms for j1=0, 4 terms for j1=1, total 8 per C block
            
            for i2 in range(2):
                for k2 in range(2):
                    # C[i1*2+i2, k1*2+k2]
                    for j1 in range(2):
                        for j2 in range(2):
                            # A[i1*2+i2, j1*2+j2] * B[j1*2+j2, k1*2+k2]
                            u_idx = (i1*2+i2) * m + (j1*2+j2)
                            v_idx = (j1*2+j2) * p + (k1*2+k2)
                            w_idx = (k1*2+k2) * n + (i1*2+i2)
                            
                            U[u_idx, col] = 1
                            V[v_idx, col] = 1
                            W[w_idx, col] = 1
                            col += 1
    
    # Verify and return
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
