"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
Uses the Kronecker product of the Strassen rank-7 decomposition for 2x2 matrices
to achieve a rank-49 decomposition for 4x4 matrix multiplication.
"""
DIMENSIONS = [4, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the tensor product of the rank-7 Strassen decomposition for 2x2.
    
    The interface requires:
    U: (n*m, rank) = (16, 49)
    V: (m*p, rank) = (16, 49)
    W: (p*n, rank) = (16, 49)
    
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid index 
    combination for the tensor, and 0 otherwise.
    Specifically, the matrix multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = 1.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen decomposition matrices.
    # We use the standard Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for 2x2 matrices:
    # U2[row, col] where row = i*2 + j (index in A), col = r (term index)
    # V2[row, col] where row = j*2 + k (index in B), col = r (term index)
    # W2[row, col] where row = k*2 + i (index in C, trace convention), col = r (term index)
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Term 0: M1 = (A00 + A11)(B00 + B11)
    # A: 00 (idx 0), 11 (idx 3)
    # B: 00 (idx 0), 11 (idx 3)
    # C: 00 (idx 0), 11 (idx 3)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1: M2 = (A10 + A11)B00
    # A: 10 (idx 2), 11 (idx 3)
    # B: 00 (idx 0)
    # C: 10 (idx 2), 11 (idx 3) -> C10 gets M2, C11 gets -M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 2: M3 = A00(B01 - B11)
    # A: 00 (idx 0)
    # B: 01 (idx 1), 11 (idx 3)
    # C: 01 (idx 1), 11 (idx 3) -> C01 gets M3, C11 gets M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 3: M4 = A11(B10 - B00)
    # A: 11 (idx 3)
    # B: 10 (idx 2), 00 (idx 0)
    # C: 00 (idx 0), 10 (idx 2) -> C00 gets M4, C10 gets M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 4: M5 = (A00 + A01)B11
    # A: 00 (idx 0), 01 (idx 1)
    # B: 11 (idx 3)
    # C: 00 (idx 0), 01 (idx 1) -> C00 gets -M5, C01 gets M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 5: M6 = (A10 - A00)(B00 + B01)
    # A: 10 (idx 2), 00 (idx 0)
    # B: 00 (idx 0), 01 (idx 1)
    # C: 11 (idx 3) -> C11 gets M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 6: M7 = (A01 - A11)(B10 + B11)
    # A: 01 (idx 1), 11 (idx 3)
    # B: 10 (idx 2), 11 (idx 3)
    # C: 00 (idx 0) -> C00 gets M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Construct 4x4 matrices using Kronecker product logic
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Vectorized construction using Kronecker products
    # For each combination of 2x2 block indices, we compute the Kronecker product
    # of the corresponding 2x2 Strassen vectors.
    
    for i in range(n):
        i1 = i // 2
        i2 = i % 2
        for j in range(m):
            j1 = j // 2
            j2 = j % 2
            u_row = i * m + j
            # The row of U is the Kronecker product of the row for (i1, j1) in U2
            # and the row for (i2, j2) in U2.
            # Note: The index in U2 for A is i*2+j.
            # So for block (i1, j1), the index is i1*2+j1.
            # For block (i2, j2), the index is i2*2+j2.
            u_val = np.kron(U2[i1 * 2 + j1], U2[i2 * 2 + j2])
            U[u_row, :] = u_val
            
    for j in range(m):
        j1 = j // 2
        j2 = j % 2
        for k in range(p):
            k1 = k // 2
            k2 = k % 2
            v_row = j * p + k
            v_val = np.kron(V2[j1 * 2 + k1], V2[j2 * 2 + k2])
            V[v_row, :] = v_val
            
    for k in range(p):
        k1 = k // 2
        k2 = k % 2
        for i in range(n):
            i1 = i // 2
            i2 = i % 2
            w_row = k * n + i
            w_val = np.kron(W2[k1 * 2 + i1], W2[k2 * 2 + i2])
            W[w_row, :] = w_val
            
    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
