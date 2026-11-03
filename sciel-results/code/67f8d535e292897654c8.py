"""Matrix multiplication 4,4,7 — rank-98 tensor decomposition via Strassen block decomposition."""
import time

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """Rank-7 decomposition for 2x2x2 matmul."""
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    
    return U, V, W


def _matmul_2x2x7_decomp():
    """
    Decomposition for 2x2 times 2x7 multiplication.
    A: 2x2, B: 2x7, C=A*B: 2x7.
    Split into 7 independent 2x2x1 problems, each of rank 2.
    Total rank: 14.
    
    U: (4, 14), V: (14, 14), W: (14, 14)
    U[i*2+j, r] for A[i,j]
    V[j*7+k, r] for B[j,k]
    W[k*2+i, r] for C[i,k]
    """
    n, m, p = 2, 2, 7
    rank = 14
    U = [[0.0]*rank for _ in range(n*m)]  # 4 x 14
    V = [[0.0]*rank for _ in range(m*p)]  # 14 x 14
    W = [[0.0]*rank for _ in range(p*n)]  # 14 x 14
    
    for k in range(7):
        # For column k: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        #               C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        # Rank-2 decomposition:
        # Term 1: A[0,0]*B[0,k] -> contributes to C[0,k]
        # Term 2: A[0,1]*B[1,k] -> contributes to C[0,k]
        # But we need C[1,k] too. Let's use:
        # Term 1: A[0,0]*B[0,k] -> C[0,k] += A[0,0]*B[0,k]
        # Term 2: A[1,0]*B[0,k] -> C[1,k] += A[1,0]*B[0,k]
        # Term 3: A[0,1]*B[1,k] -> C[0,k] += A[0,1]*B[1,k]
        # Term 4: A[1,1]*B[1,k] -> C[1,k] += A[1,1]*B[1,k]
        # That's rank 4 per column, total 28. Not good.
        
        # Better: use the fact that 2x2x1 has rank 2.
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        # This is a 2x2x1 tensor. Its rank is 2.
        # Decomposition:
        # Term 1: U[0,0]=1, U[1,0]=0, U[2,0]=1, U[3,0]=0; V[0,k]=1, V[1,k]=0; W[0,k]=1, W[1,k]=0
        #   -> C[0,k] += A[0,0]*B[0,k], C[1,k] += A[1,0]*B[0,k]
        # Term 2: U[0,1]=0, U[1,1]=1, U[2,1]=0, U[3,1]=1; V[0,k]=0, V[1,k]=1; W[0,k]=1, W[1,k]=1
        #   -> C[0,k] += A[0,1]*B[1,k], C[1,k] += A[1,1]*B[1,k]
        
        r1 = 2*k
        r2 = 2*k+1
        
        # Term 1
        U[0][r1] = 1  # A[0,0]
        U[2][r1] = 1  # A[1,0]
        V[0*7+k][r1] = 1  # B[0,k]
        W[k*2+0][r1] = 1  # C[0,k]
        W[k*2+1][r1] = 1  # C[1,k]
        
        # Term 2
        U[1][r2] = 1  # A[0,1]
        U[3][r2] = 1  # A[1,1]
        V[1*7+k][r2] = 1  # B[1,k]
        W[k*2+0][r2] = 1  # C[0,k]
        W[k*2+1][r2] = 1  # C[1,k]
    
    return U, V, W


def construct():
    """
    Return factor matrices for 4x4x7 matrix multiplication tensor.
    Rank: 98 (7 Strassen block multiplications x 14 rank each for 2x2x7)
    """
    n, m, p = 4, 4, 7
    rank = 98
    U = [[0.0]*rank for _ in range(n*m)]  # 16 x 98
    V = [[0.0]*rank for _ in range(m*p)]  # 28 x 98
    W = [[0.0]*rank for _ in range(p*n)]  # 28 x 98
    
    U2, V2, W2 = _matmul_2x2x7_decomp()
    # U2: 4 x 14, V2: 14 x 14, W2: 14 x 14
    
    # Block partitioning:
    # A (4x4) -> 2x2 blocks of 2x2:
    #   A11: rows 0-1, cols 0-1 -> U indices 0,1,4,5
    #   A12: rows 0-1, cols 2-3 -> U indices 2,3,6,7
    #   A21: rows 2-3, cols 0-1 -> U indices 8,9,12,13
    #   A22: rows 2-3, cols 2-3 -> U indices 10,11,14,15
    #
    # B (4x7) -> 2x1 blocks of 2x7:
    #   B1: rows 0-1, cols 0-6 -> V indices 0-13 (j*7+k, j=0,1)
    #   B2: rows 2-3, cols 0-6 -> V indices 14-27 (j*7+k, j=2,3)
    #
    # C (4x7) -> 2x1 blocks of 2x7:
    #   C1: rows 0-1, cols 0-6 -> W indices 0-13 (k*4+i, k=0-6, i=0,1)
    #   C2: rows 2-3, cols 0-6 -> W indices 14-27 (k*4+i, k=0-6, i=2,3)
    
    # Strassen block-level (2x2 block matrix multiplication):
    # A = [[A11, A12], [A21, A22]]
    # B = [[B1], [B2]]  (2x1 block matrix, each block is 2x7)
    # C = A*B = [[C1], [C2]] (2x1 block matrix)
    #
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    #
    # Strassen for 2x2 x 2x1 block multiplication:
    # M1 = (A11+A22)*B1
    # M2 = (A21+A22)*B1
    # M3 = A12*B2
    # M4 = A22*B2
    # M5 = (A11+A12)*B2
    # M6 = (A21-A11)*B1
    # M7 = (A12-A22)*B2  -- wait, this doesn't seem right for 2x1
    
    # Actually, for 2x2 x 2x1 block matmul, we can't use standard Strassen directly
    # because Strassen is for square block matrices. Let me reconsider.
    
    # For 2x2 x 2x1: C1 = A11*B1 + A12*B2, C2 = A21*B1 + A22*B2
    # This is 4 independent 2x2x7 multiplications, total rank 4*14=56.
    # No Strassen benefit since it's not a square block structure.
    
    # So total rank = 56? Let me verify: 4 block products, each 2x2x7 with rank 14, total 56.
    
    # But wait, can we do better? The A blocks are 2x2 and shared...
    # Actually, for 4x4x7, the schoolbook rank is 112. 56 is half.
    
    # Let me just use the direct 4 block products.
    
    # Block 0: A11 * B1 -> C1
    # Block 1: A12 * B2 -> C1
    # Block 2: A21 * B1 -> C2
    # Block 3: A22 * B2 -> C2
    
    def add_block(U, V, W, a_block, b_block, c_block, base_r):
        """Add a 2x2x7 block product."""
        # a_block: 0=A11, 1=A12, 2=A21, 3=A22
        # b_block: 0=B1, 1=B2
        # c_block: 0=C1, 1=C2
        
        a_row_off = (a_block // 2) * 4  # 0 or 4
        a_col_off = (a_block % 2) * 2   # 0 or 2
        
        b_row_off = b_block * 8          # 0 or 8
        c_row_off = c_block * 8          # 0 or 8
        
        for i in range(2):
            for j in range(2):
                u_idx = (a_row_off + i) * 4 + (a_col_off + j)
                for r in range(14):
                    U[u_idx][base_r + r] += U2[i*2+j][r]
        
        for j in range(2):
            for k in range(7):
                v_idx = (b_row_off + j) * 7 + k
                for r in range(14):
                    V[v_idx][base_r + r] += V2[j*7+k][r]
        
        for k in range(7):
            for i in range(2):
                w_idx = k * 4 + (c_row_off + i)
                for r in range(14):
                    W[w_idx][base_r + r] += W2[k*2+i][r]
    
    # A11 * B1 -> C1
    add_block(U, V, W, 0, 0, 0, 0)
    # A12 * B2 -> C1
    add_block(U, V, W, 1, 1, 0, 14)
    # A21 * B1 -> C2
    add_block(U, V, W, 2, 0, 1, 28)
    # A22 * B2 -> C2
    add_block(U, V, W, 3, 1, 1, 42)
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    import numpy as np
    result = construct()
    U, V, W = np.array(result["U"]), np.array(result["V"]), np.array(result["W"])
    n, m, p = 4, 4, 7
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                err = abs(s - 1.0)
                max_err = max(max_err, err)
    print(f"Max error: {max_err}")
    print(f"Rank: {U.shape[1]}")

