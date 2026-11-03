"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the tensor (Kronecker) product of the rank-7 Strassen decomposition for 2x2.
    Includes a permutation to fix the index ordering for the official trace convention.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    n, m, p = DIMENSIONS

    # Define the 2x2 Strassen decomposition matrices
    # Convention:
    # U has shape (n*m, rank) -> for 2x2, (4, 7). Index: i*m + j
    # V has shape (m*p, rank) -> for 2x2, (4, 7). Index: j*p + k
    # W has shape (p*n, rank) -> for 2x2, (4, 7). Index: k*n + i
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For 2x2, T[i,j,k] should be 1 if i==k else 0.
    
    # Strassen 2x2 matrices (standard formulation):
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # M1 = (A10+A11)B00       -> C10 += M1
    # M2 = A00(B10-B11)       -> C01 += M2
    # M3 = A11(B01-B00)       -> C01 -= M3, C11 += M3
    # M4 = (A00+A01)B11       -> C00 -= M4, C01 += M4
    # M5 = (A10-A00)(B00+B01) -> C10 -= M5
    # M6 = (A01-A11)(B10+B11) -> C10 += M6
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M0 = (A00+A11)(B00+B11)
    # A00: i=0,j=0 -> U[0]
    # A11: i=1,j=1 -> U[3]
    # B00: j=0,k=0 -> V[0]
    # B11: j=1,k=1 -> V[3]
    # C00: i=0,k=0 -> W[0*2+0]=0
    # C11: i=1,k=1 -> W[1*2+1]=3
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M1 = (A10+A11)B00
    # A10: i=1,j=0 -> U[2]
    # A11: i=1,j=1 -> U[3]
    # B00: j=0,k=0 -> V[0]
    # C10: i=1,k=0 -> W[0*2+1]=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1
    
    # M2 = A00(B10-B11)
    # A00: i=0,j=0 -> U[0]
    # B10: j=1,k=0 -> V[2]
    # B11: j=1,k=1 -> V[3]
    # C01: i=0,k=1 -> W[1*2+0]=2
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1
    
    # M3 = A11(B01-B00)
    # A11: i=1,j=1 -> U[3]
    # B01: j=0,k=1 -> V[1]
    # B00: j=0,k=0 -> V[0]
    # C01: i=0,k=1 -> W[2]
    # C11: i=1,k=1 -> W[3]
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = 1
    
    # M4 = (A00+A01)B11
    # A00: i=0,j=0 -> U[0]
    # A01: i=0,j=1 -> U[1]
    # B11: j=1,k=1 -> V[3]
    # C00: i=0,k=0 -> W[0]
    # C01: i=0,k=1 -> W[2]
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M5 = (A10-A00)(B00+B01)
    # A10: i=1,j=0 -> U[2]
    # A00: i=0,j=0 -> U[0]
    # B00: j=0,k=0 -> V[0]
    # B01: j=0,k=1 -> V[1]
    # C10: i=1,k=0 -> W[1]
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[1, 5] = -1
    
    # M6 = (A01-A11)(B10+B11)
    # A01: i=0,j=1 -> U[1]
    # A11: i=1,j=1 -> U[3]
    # B10: j=1,k=0 -> V[2]
    # B11: j=1,k=1 -> V[3]
    # C10: i=1,k=0 -> W[1]
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1
    
    # Verify 2x2 Strassen
    # T[i,j,k] = sum_c U[i*2+j, c] V[j*2+k, c] W[k*2+i, c]
    # Should be 1 if i==k else 0.
    
    # Now construct 4x4 via Kronecker product.
    # The Kronecker product of two decompositions gives a decomposition for the block matrix product.
    # However, the index ordering must match.
    # Standard Kronecker product of U2, V2, W2:
    # U4 = np.kron(U2, U2)
    # This works if the indices are ordered consistently.
    # U2 index: i*2+j.
    # U4 index: i*4+j.
    # If we split i = i1*2+i2 and j = j1*2+j2.
    # Then i*4+j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2.
    # np.kron(U2, U2) index: (i1*2+j1)*4 + (i2*2+j2) = i1*8 + j1*4 + i2*2 + j2.
    # These are different.
    # We need a permutation.
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Permutation for U4:
    # Row index S in 0..15.
    # S = i*4+j.
    # i = S//4, j = S%4.
    # i1 = i//2, i2 = i%2.
    # j1 = j//2, j2 = j%2.
    # Kron index K = (i1*2+j1)*4 + (i2*2+j2).
    
    perm_U = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i1 = i // 2
        i2 = i % 2
        j1 = j // 2
        j2 = j % 2
        K = (i1 * 2 + j1) * 4 + (i2 * 2 + j2)
        perm_U[S] = K
        
    # Permutation for V4:
    # Row index S in 0..15.
    # S = j*4+k.
    # j = S//4, k = S%4.
    # j1 = j//2, j2 = j%2.
    # k1 = k//2, k2 = k%2.
    # Kron index K = (j1*2+k1)*4 + (j2*2+k2).
    
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j1 = j // 2
        j2 = j % 2
        k1 = k // 2
        k2 = k % 2
        K = (j1 * 2 + k1) * 4 + (j2 * 2 + k2)
        perm_V[S] = K
        
    # Permutation for W4:
    # Row index S in 0..15.
    # S = k*4+i.
    # k = S//4, i = S%4.
    # k1 = k//2, k2 = k%2.
    # i1 = i//2, i2 = i%2.
    # Kron index K = (k1*2+i1)*4 + (k2*2+i2).
    
    perm_W = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k1 = k // 2
        k2 = k % 2
        i1 = i // 2
        i2 = i % 2
        K = (k1 * 2 + i1) * 4 + (k2 * 2 + i2)
        perm_W[S] = K
        
    U4_fixed = U4[perm_U, :]
    V4_fixed = V4[perm_V, :]
    W4_fixed = W4[perm_W, :]
    
    if time.time() - start_time > 110:
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
# EVOLVE-BLOCK-END
