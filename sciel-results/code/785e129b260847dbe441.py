"""Structural tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Strassen decomposition for 2x2 * 2x2 (rank 7)
    # Returns U, V, W each of shape (4, 7)
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        # M3 = A11(B21-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        # M4 = (A11-A22)(B22+B33) -- wait, for 2x2 it's (A11-A22)(B12+B22)
        # Actually for 2x2: M4 = (A11-A22)(B12+B22)
        U[0, 3] = 1; U[3, 3] = -1
        V[1, 3] = 1; V[3, 3] = 1
        W[0, 3] = 1; W[3, 3] = 1
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[0, 5] = 1
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[2, 6] = 1
        return U, V, W

    # Build 4x4 * 4x4 using Strassen recursively (rank 49)
    # Actually, let's just do block decomposition directly.
    # For 4x4 * 4x4: split into 2x2 blocks of size 2.
    # That gives 4 independent 2x2*2x2 problems, each rank 7 = 28? No.
    # Better: use the recursive structure. 4x4*4x4 can be done with rank 49 using
    # the known optimal decomposition for 4x4*4x4.
    # Actually, the best known for 4x4*4x4 is rank 49 (Le Gall 1981).
    # Let me construct it via 2-level Strassen: rank = 7^2 = 49.
    
    def strassen_4x4():
        U2, V2, W2 = strassen_2x2()
        # U for 4x4*4x4: shape (16, 49)
        # Block (bi, bj) where bi,bj in {0,1}
        # For each Strassen term t in 0..6 for the "top" level,
        # and Strassen term s in 0..6 for the "inner" level
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        for t in range(7):
            for s in range(7):
                col = t * 7 + s
                # U: rows are (i,j) for 4x4 matrix, i in 0..3, j in 0..3
                # Block row = i//2, inner row = i%2
                # For top-level Strassen term t, we need U2[block_row, t]
                # For inner-level Strassen term s, we need U2[inner_row, s]
                # Actually this is getting complex. Let me use a different approach.
                pass
        # This is too complex to construct manually. Let me use a simpler approach.
        # For 4x4*4x4, use the "dumb" block decomposition:
        # Split into 4 blocks of 2x2. Each block multiplication is independent.
        # That gives rank 4*7 = 28? No, that's not right either.
        # Actually, if we split A into 2x2 blocks (4 blocks), B into 2x2 blocks,
        # each C block depends on multiple A and B blocks.
        
        # Let me just use the Kronecker product approach for Strassen.
        # The rank-49 decomposition for 4x4*4x4 can be built as follows:
        # Use 2-level Strassen. The indices are (t1, t2) where t1, t2 in 0..6.
        # For the outer level, the "matrices" are 2x2 blocks.
        # For the inner level, each entry of those 2x2 blocks is a scalar.
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        for t1 in range(7):
            for t2 in range(7):
                col = t1 * 7 + t2
                # Outer level: U2[i1, t1] where i1 is block row (0 or 1)
                # Inner level: U2[i2, t2] where i2 is row within block (0 or 1)
                # Full row index = i1*2 + i2
                for i1 in range(2):
                    if U2[i1, t1] != 0:
                        for i2 in range(2):
                            if U2[i2, t2] != 0:
                                row = i1 * 2 + i2
                                U[row, col] = U2[i1, t1] * U2[i2, t2]
                for i1 in range(2):
                    if V2[i1, t1] != 0:
                        for i2 in range(2):
                            if V2[i2, t2] != 0:
                                row = i1 * 2 + i2
                                V[row, col] = V2[i1, t1] * V2[i2, t2]
                for i1 in range(2):
                    if W2[i1, t1] != 0:
                        for i2 in range(2):
                            if W2[i2, t2] != 0:
                                row = i1 * 2 + i2
                                W[row, col] = W2[i1, t1] * W2[i2, t2]
        
        return U, V, W

    # Verify the 4x4*4x4 decomposition
    def verify_decomp(U, V, W, n, m, p):
        """U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Build 4x4 * 4x5 decomposition
    # Split the 4x5 matrix into a 4x4 part and a 4x1 part.
    # The 4x4 part uses rank 49, the 4x1 part uses rank 4.
    # Total rank = 53.
    
    U44, V44, W44 = strassen_4x4()
    
    # 4x4 * 4x1 part: rank 4
    U41 = np.zeros((16, 4))
    V41 = np.zeros((4, 4))
    W41 = np.zeros((4, 4))
    for col in range(4):
        for j in range(4):
            U41[j * 4 + col, col] = 1.0  # A[j, col]
            V41[col, col] = 1.0  # B[col, 0]
            W41[col, col] = 1.0  # C[col, 0]
    
    # Combine: for 4x4 * 4x5
    # U: (16, 53) - same as U44 for first 49 cols, U41 for next 4
    # V: (20, 53) - V44 for first 16 rows and 49 cols, V41 for next 4 rows and 4 cols
    # W: (20, 53) - W44 for first 16 rows and 49 cols, W41 for next 4 rows and 4 cols
    
    U = np.zeros((16, 53))
    V = np.zeros((20, 53))
    W = np.zeros((20, 53))
    
    U[:, :49] = U44
    U[:, 49:] = U41
    
    V[:16, :49] = V44
    V[16:20, 49:] = V41
    
    W[:16, :49] = W44
    W[16:20, 49:] = W41
    
    # Verify
    if not verify_decomp(U, V, W, 4, 4, 5):
        # Fallback to schoolbook
        rank = 80
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j, col] = 1.0
                    V[j*5+k, col] = 1.0
                    W[k*4+i, col] = 1.0
                    col += 1
    else:
        rank = 53

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
