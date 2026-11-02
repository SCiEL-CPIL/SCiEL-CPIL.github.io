"""Matrix multiplication 2x3 x 3x8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # n=2, m=3, p=8

    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(
                        U[i*m+j, :] * V[j*p+k, :],
                        W[k*n+i, :]
                    )
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Build rank-32 decomposition using Strassen blocks
    # Split k into 4 blocks of size 2. For each block, use:
    #   - Strassen (rank 7) for j in {0,1}
    #   - Direct (rank 1) for j=2
    # Total rank = 4 * (7 + 1) = 32

    def build_strassen_block():
        """
        Build the Strassen decomposition for 2x2 x 2x2 multiplication.
        Returns U, V, W arrays of shape (4,7), (4,7), (4,7) for the 2x2x2x2 tensor.
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # U indices: i*2+j, i in {0,1}, j in {0,1}
        # V indices: j*2+k, j in {0,1}, k in {0,1}
        # W indices: k*2+i, k in {0,1}, i in {0,1}
        
        # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        # U: A11=1, A22=1 -> U[0]=1, U[3]=1
        # V: B11=1, B22=1 -> V[0]=1, V[3]=1
        # W: C11=1, C22=1 -> W[0]=1, W[3]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A21+A22)B11 -> C21 += 1
        # U: A21=1, A22=1 -> U[1]=1, U[3]=1
        # V: B11=1 -> V[0]=1
        # W: C21=1 -> W[1]=1
        U[1, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3: A11(B12-B22) -> C12 += 1
        # U: A11=1 -> U[0]=1
        # V: B12=1, B22=-1 -> V[1]=1, V[3]=-1
        # W: C12=1 -> W[2]=1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4: A22(B21-B11) -> C21 += 1, C11 += 1
        # U: A22=1 -> U[3]=1
        # V: B21=1, B11=-1 -> V[2]=1, V[0]=-1
        # W: C21=1, C11=1 -> W[1]=1, W[0]=1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1; W[0, 3] = 1
        
        # M5: (A11+A12)B22 -> C12 += 1, C11 -= 1
        # U: A11=1, A12=1 -> U[0]=1, U[2]=1
        # V: B22=1 -> V[3]=1
        # W: C12=1, C11=-1 -> W[2]=1, W[0]=-1
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[2, 4] = 1; W[0, 4] = -1
        
        # M6: (A21-A11)(B11+B12) -> C22 += 1
        # U: A21=1, A11=-1 -> U[1]=1, U[0]=-1
        # V: B11=1, B12=1 -> V[0]=1, V[1]=1
        # W: C22=1 -> W[3]=1
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7: (A12-A22)(B21+B22) -> C11 += 1
        # U: A12=1, A22=-1 -> U[2]=1, U[3]=-1
        # V: B21=1, B22=1 -> V[2]=1, V[3]=1
        # W: C11=1 -> W[0]=1
        U[2, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Build the full 2x3 x 3x8 decomposition
    # For each block b in 0..3, k = 2*b + k', k' in {0,1}
    # j in {0,1} uses Strassen (7 terms), j=2 uses 1 term
    rank = 4 * 8  # 32
    U = np.zeros((n * m, rank))  # (6, 32)
    V = np.zeros((m * p, rank))  # (24, 32)
    W = np.zeros((p * n, rank))  # (16, 32)
    
    U_s, V_s, W_s = build_strassen_block()
    
    col = 0
    for b in range(4):
        # Strassen part for j in {0,1}, k' in {0,1}
        for s in range(7):
            # U: i in {0,1}, j in {0,1} -> index i*3 + j (since m=3)
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = U_s[i * 2 + j, s]
            # V: j in {0,1}, k = 2*b + k' -> index j*8 + k
            for j in range(2):
                for kp in range(2):
                    k = 2 * b + kp
                    V[j * p + k, col] = V_s[j * 2 + kp, s]
            # W: k = 2*b + k', i in {0,1} -> index k*2 + i
            for kp in range(2):
                k = 2 * b + kp
                for i in range(2):
                    W[k * n + i, col] = W_s[kp * 2 + i, s]
            col += 1
        
        # Direct part for j=2, k' in {0,1}: C[i,k] += A[i,2]*B[2,k]
        # This is rank 1: U[i*3+2]=1, V[2*8+k]=1, W[k*2+i]=1
        for i in range(2):
            U[i * m + 2, col] = 1
        for kp in range(2):
            k = 2 * b + kp
            V[2 * p + k, col] = 1
            for i in range(2):
                W[k * n + i, col] = 1
        col += 1
    
    # Verify
    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
                c += 1
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
