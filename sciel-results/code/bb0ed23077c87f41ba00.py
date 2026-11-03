import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 69 decomposition:
    # 49 terms for 4x4x4 part (Strassen x Strassen)
    # 20 terms for 4x1x5 part (trivial)
    
    R = 69
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # --- Part 1: 4x4x4 decomposition ---
    # Indices for this part:
    # A: i in 0..3, j in 0..3
    # B: j in 0..3, k in 0..3
    # C: k in 0..3, i in 0..3
    
    # Get 2x2x2 Strassen decomposition
    # Returns U2, V2, W2 of shape (4, 7)
    def get_strassen_2x2():
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # Term 2: (A10+A11)B00 -> C10, C11
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[1, 1] = 1; W2[3, 1] = 1
        
        # Term 3: A00(B01-B11) -> C01
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1
        
        # Term 4: A11(B10-B00) -> C00, C10
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[1, 3] = 1
        
        # Term 5: (A00+A01)B11 -> C00, C11
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[3, 4] = 1
        
        # Term 6: (A10-A00)(B00+B01) -> C11
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        # Term 7: (A01-A11)(B10+B11) -> C00
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        return U2, V2, W2

    U2, V2, W2 = get_strassen_2x2()
    
    # Construct 4x4x4 decomposition using tensor product of 2x2x2 decompositions
    # We iterate over the 7 terms of the "outer" 2x2x2 and 7 terms of the "inner" 2x2x2
    # This effectively computes the rank 49 decomposition
    
    idx = 0
    for r1 in range(7):
        for r2 in range(7):
            # u = u1 x u2, v = v1 x v2, w = w1 x w2
            # Need to map indices correctly
            # u1 is in R^4 (indices 0..3 for 2x2 matrix A)
            # u2 is in R^4
            # u4 is in R^16
            
            u4 = np.kron(U2[:, r1], U2[:, r2])
            v4 = np.kron(V2[:, r1], V2[:, r2])
            w4 = np.kron(W2[:, r1], W2[:, r2])
            
            # Map u4 (size 16) to U (size 20)
            # u4 indices: i_block * 2 + i_local
            # U indices: i * 5 + j
            # i_block is i, i_local is j?
            # In 2x2x2, index is i*2+j.
            # In 4x4x4, we are splitting i into i1, i0.
            # i = 2*i1 + i0.
            # j = 2*j1 + j0.
            # The tensor product index is (i1, i0, j1, j0).
            # np.kron produces index (i1*2 + i0) * 2 + (j1*2 + j0)?
            # No, np.kron(A, B) where A is 4x1, B is 4x1 produces 16x1.
            # Index k in result corresponds to i_A * 4 + i_B.
            # i_A is index in U2 (0..3), i_B is index in U2 (0..3).
            # i_A = i1*2 + j1? No.
            # U2 index is i*2+j.
            # So i_A = 2*i1 + j1? No, i_A is the row index of the 2x2 matrix.
            # Let's say U2[i_A] corresponds to A[i1, j1] where i_A = 2*i1 + j1.
            # Then u4[i_A * 4 + i_B] = U2[i_A] * U2[i_B].
            # i_A = 2*i1 + j1, i_B = 2*i0 + j0.
            # So u4 index = (2*i1 + j1)*4 + (2*i0 + j0) = 8*i1 + 4*j1 + 2*i0 + j0.
            
            # We need to place this into U[i*5 + j] where i = 2*i1 + i0, j = 2*j1 + j0.
            # U index = (2*i1 + i0)*5 + (2*j1 + j0) = 10*i1 + 5*i0 + 2*j1 + j0.
            
            # We can just iterate over the 16 non-zero entries of u4
            for k in range(16):
                val = u4[k]
                if val != 0:
                    i_A = k // 4
                    i_B = k % 4
                    i1 = i_A // 2
                    j1 = i_A % 2
                    i0 = i_B // 2
                    j0 = i_B % 2
                    
                    i_global = 2 * i1 + i0
                    j_global = 2 * j1 + j0
                    
                    U[i_global * 5 + j_global, idx] = val
            
            # Similarly for V
            # V2 index is j*2+k.
            # v4 index = (2*j1 + k1)*4 + (2*j0 + k0).
            # V index = j*5 + k = (2*j1 + j0)*5 + (2*k1 + k0) = 10*j1 + 5*j0 + 2*k1 + k0.
            
            for k in range(16):
                val = v4[k]
                if val != 0:
                    j_A = k // 4
                    k_A = k % 4
                    j1 = j_A // 2
                    k1 = j_A % 2
                    j0 = k_A // 2
                    k0 = k_A % 2
                    
                    j_global = 2 * j1 + j0
                    k_global = 2 * k1 + k0
                    
                    V[j_global * 5 + k_global, idx] = val
            
            # Similarly for W
            # W2 index is k*2+i.
            # w4 index = (2*k1 + i1)*4 + (2*k0 + i0).
            # W index = k*4 + i = (2*k1 + k0)*4 + (2*i1 + i0) = 8*k1 + 4*k0 + 2*i1 + i0.
            
            for k in range(16):
                val = w4[k]
                if val != 0:
                    k_A = k // 4
                    i_A = k % 4
                    k1 = k_A // 2
                    i1 = k_A % 2
                    k0 = i_A // 2
                    i0 = i_A % 2
                    
                    k_global = 2 * k1 + k0
                    i_global = 2 * i1 + i0
                    
                    W[k_global * 4 + i_global, idx] = val
            
            idx += 1
    
    # --- Part 2: 4x1x5 decomposition ---
    # A: i in 0..3, j=4
    # B: j=4, k in 0..4
    # C: k in 0..4, i in 0..3
    
    for i in range(4):
        for k in range(5):
            # U index: i*5 + 4
            # V index: 4*5 + k = 20 + k
            # W index: k*4 + i
            
            U[i * 5 + 4, idx] = 1.0
            V[20 + k, idx] = 1.0
            W[k * 4 + i, idx] = 1.0
            
            idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
