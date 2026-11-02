import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # The rank of M_{2,2,8} is 28 (4 blocks of M_{2,2,2}, each rank 7)
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # A is 2x2, B is 2x2, C = AB is 2x2
    # Indices: A_{i,j}, B_{j,k}, C_{i,k}
    # U corresponds to A, V to B, W to C
    # Term t: U[:,t], V[:,t], W[:,t]
    
    # Strassen terms definition:
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00     -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11)     -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00)     -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11     -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7
    
    # We define the coefficients for U, V, W for each of the 7 terms.
    # U index: i*m + j (row i, col j of A)
    # V index: j*p + k (row j, col k of B) -- Note: k is local 0..1 for the block
    # W index: k*n + i (row k, col i of C) -- Note: k is local 0..1 for the block
    
    # The 4 blocks of B correspond to columns (0,1), (2,3), (4,5), (6,7)
    # Let's iterate over the 4 blocks q = 0..3
    # For each block, we apply the 7 Strassen terms.
    
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, 
         {(0,0): 1}, 
         {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, 
         {(0,1): 1, (1,1): -1}, 
         {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, 
         {(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, 
         {(1,1): 1}, 
         {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (0,1): 1}, 
         {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, 
         {(1,0): 1, (1,1): 1}, 
         {(0,0): 1}),
    ]
    
    for q in range(4):
        k_base = 2 * q
        for r in range(7):
            t = q * 7 + r
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[r]
            
            # Fill U
            for (i, j), coeff in u_coeffs.items():
                U[i * m + j, t] = coeff
            
            # Fill V
            # j is row of B (0 or 1)
            # k_local is col of B within block (0 or 1)
            # Global col k = k_base + k_local
            for (j, k_local), coeff in v_coeffs.items():
                k_global = k_base + k_local
                V[j * p + k_global, t] = coeff
            
            # Fill W
            # i is col of C (0 or 1)
            # k_local is row of C within block (0 or 1)
            # Global row k = k_base + k_local
            # W index is k_global * n + i
            for (i, k_local), coeff in w_coeffs.items():
                k_global = k_base + k_local
                W[k_global * n + i, t] = coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
