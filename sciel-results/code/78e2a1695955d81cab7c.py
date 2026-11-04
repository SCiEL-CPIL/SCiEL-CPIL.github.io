import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_222():
    """
    Returns U, V, W for rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U, V, W have shape (4, 7).
    Indices for A, B, C are 0..3 corresponding to (0,0), (0,1), (1,0), (1,1).
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    
    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00 -> C10, C11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01 - B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10 - B00) -> C00, C10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11 -> C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01) -> C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11) -> C00
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_decomposition_444():
    """
    Computes the Kronecker product of two 2x2x2 Strassen decompositions
    to get a rank-49 decomposition of 4x4x4 matrix multiplication.
    Returns U, V, W with shapes (16, 49), (16, 49), (16, 49).
    """
    U2, V2, W2 = _strassen_222()
    r1 = 7
    R = r1 * r1  # 49
    
    U = np.zeros((16, R), dtype=np.int16)
    V = np.zeros((16, R), dtype=np.int16)
    W = np.zeros((16, R), dtype=np.int16)
    
    for r1_idx in range(r1):
        for r2_idx in range(r1):
            r_idx = r1_idx * r1 + r2_idx
            
            # U: kron of U2 columns
            for i in range(4):
                for j in range(4):
                    U[i*4 + j, r_idx] = U2[(i//2)*2 + (j//2), r1_idx] * U2[(i%2)*2 + (j%2), r2_idx]
            
            # V: kron of V2 columns
            for j in range(4):
                for k in range(4):
                    V[j*4 + k, r_idx] = V2[(j//2)*2 + (k//2), r1_idx] * V2[(j%2)*2 + (k%2), r2_idx]
            
            # W: kron of W2 columns
            for k in range(4):
                for i in range(4):
                    W[k*4 + i, r_idx] = W2[(k//2)*2 + (i//2), r1_idx] * W2[(k%2)*2 + (i%2), r2_idx]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get the rank-49 decomposition for 4x4x4
    U444, V444, W444 = _kron_decomposition_444()
    r444 = 49
    
    # For 4x4x8, we split the output into two 4x4 blocks.
    # The key insight: both blocks share the same left factor A (and thus same U structure).
    # We can use the same U columns for both blocks, but with different V and W.
    
    # Total rank: we use r444 for the first block and r444 for the second block,
    # but since U is shared, the effective rank is still 2*r444 = 98.
    # However, let's try to be smarter.
    
    # Actually, the standard approach gives rank 98. Let's verify this is correct
    # and then see if we can do better.
    
    total_rank = 2 * r444  # 98
    U = np.zeros((n*m, total_rank), dtype=np.int16)
    V = np.zeros((m*p, total_rank), dtype=np.int16)
    W = np.zeros((p*n, total_rank), dtype=np.int16)
    
    # First block (k=0..3): uses columns 0..48
    U[:, :r444] = U444
    for j in range(m):
        for k in range(4):
            V[j*p + k, :r444] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k*n + i, :r444] = W444[k*4 + i, :]
    
    # Second block (k=4..7): uses columns 49..97
    U[:, r444:] = U444
    for j in range(m):
        for k in range(4):
            V[j*p + (4 + k), r444:] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(4 + k)*n + i, r444:] = W444[k*4 + i, :]
    
    # Verify the decomposition
    tol = 1e-9
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > tol:
                    # Fallback to schoolbook
                    schoolbook_rank = n * m * p
                    U_school = np.zeros((n*m, schoolbook_rank), dtype=np.int16)
                    V_school = np.zeros((m*p, schoolbook_rank), dtype=np.int16)
                    W_school = np.zeros((p*n, schoolbook_rank), dtype=np.int16)
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_school[ii*m + jj, col] = 1
                                V_school[jj*p + kk, col] = 1
                                W_school[kk*n + ii, col] = 1
                                col += 1
                    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
