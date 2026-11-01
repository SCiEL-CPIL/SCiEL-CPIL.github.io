import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds safety margin
    
    # Strassen's 7-term algorithm for 2x2 matrix multiplication.
    # We group the 8 columns of B into 4 pairs.
    # Each pair is a 2x2 sub-matrix.
    # We apply Strassen to each pair independently.
    # Total rank: 4 * 7 = 28.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen products definitions
    # A_coeffs: {(i,j): coeff}
    # B_coeffs: {(j,k_sub): coeff}
    # C_contrib: {(i, k_sub): coeff}
    strassen_products = [
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        {'A': {(0,0): 1, (1,1): 1},
         'B': {(0,0): 1, (1,1): 1},
         'C': {(0,0): 1, (1,1): 1}},
        # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
        {'A': {(1,0): 1, (1,1): 1},
         'B': {(0,0): 1},
         'C': {(1,0): 1, (1,1): -1}},
        # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
        {'A': {(0,0): 1},
         'B': {(0,1): 1, (1,1): -1},
         'C': {(0,1): 1, (1,1): 1}},
        # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
        {'A': {(1,1): 1},
         'B': {(1,0): 1, (0,0): -1},
         'C': {(0,0): 1, (1,0): 1}},
        # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
        {'A': {(0,0): 1, (0,1): 1},
         'B': {(1,1): 1},
         'C': {(0,0): -1, (0,1): 1}},
        # M6 = (A10-A00)(B00+B01) -> C11+=M6
        {'A': {(1,0): 1, (0,0): -1},
         'B': {(0,0): 1, (0,1): 1},
         'C': {(1,1): 1}},
        # M7 = (A01-A11)(B10+B11) -> C00+=M7
        {'A': {(0,1): 1, (1,1): -1},
         'B': {(1,0): 1, (1,1): 1},
         'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r
            
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    # Verify the decomposition is correct
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # C[i,k] = sum_j T[i,j,k]
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    C_computed = np.einsum('ijk->ik', T)
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = i * m * p + j * p + k
                    U[i * m + j, t] = 1
                    V[j * p + k, t] = 1
                    W[k * n + i, t] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check if time is running out
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt to find linear dependencies to reduce rank below 28
    # The tensor is M_{2,2,8} = M_{2,2,2} ⊗ I_4 (in some sense)
    # We have 4 copies of the 7-term Strassen decomposition.
    # Check if there are linear dependencies among the 28 rank-1 terms.
    
    # Build the 28x28 Gram matrix for each of U, V, W
    # If any Gram matrix has rank < 28, we can reduce the decomposition.
    
    # Actually, for a tensor decomposition, linear dependencies in the U columns
    # don't directly imply we can reduce rank. We need to check if the tensor
    # itself has lower rank.
    
    # Let's try a different approach: check if we can express some columns
    # as linear combinations of others while preserving the tensor identity.
    
    # For now, return the verified rank-28 decomposition.
    # Future work: implement a search for rank reductions.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
