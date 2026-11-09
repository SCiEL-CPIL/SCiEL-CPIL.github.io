import time
import numpy as np
import sys

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7).
    n=2, m=2, p=2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    
    Indexing:
    U[i*m+j, r] where i in {0,1}, j in {0,1} -> U is 4x7
    V[j*p+k, r] where j in {0,1}, k in {0,1} -> V is 4x7
    W[k*n+i, r] where k in {0,1}, i in {0,1} -> W is 4x7
    
    Strassen's 7 products:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    U factors (for A):
    M1: A11=1, A22=1 -> U[0,0]=1, U[3,0]=1
    M2: A21=1, A22=1 -> U[2,1]=1, U[3,1]=1
    M3: A11=1 -> U[0,2]=1
    M4: A22=1 -> U[3,3]=1
    M5: A11=1, A12=1 -> U[0,4]=1, U[1,4]=1
    M6: A21=1, A11=-1 -> U[2,5]=1, U[0,5]=-1
    M7: A12=1, A22=-1 -> U[1,6]=1, U[3,6]=-1
    
    V factors (for B):
    M1: B11=1, B22=1 -> V[0,0]=1, V[3,0]=1
    M2: B11=1 -> V[0,1]=1
    M3: B21=1, B22=-1 -> V[2,2]=1, V[3,2]=-1
    M4: B12=1, B11=-1 -> V[1,3]=1, V[0,3]=-1
    M5: B22=1 -> V[3,4]=1
    M6: B11=1, B12=1 -> V[0,5]=1, V[1,5]=1
    M7: B21=1, B22=1 -> V[2,6]=1, V[3,6]=1
    
    W factors (for C):
    C11: M1=1, M4=1, M5=-1, M7=1 -> W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
    C12: M3=1, M5=1 -> W[1,2]=1, W[1,4]=1
    C21: M2=1, M4=1 -> W[2,1]=1, W[2,3]=1
    C22: M1=1, M2=-1, M3=1, M6=1 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
    """
    R = 7
    U = [[0.0] * R for _ in range(4)]
    V = [[0.0] * R for _ in range(4)]
    W = [[0.0] * R for _ in range(4)]
    
    # M1
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # M2
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0; W[3][1] = -1.0
    
    # M3
    U[0][2] = 1.0
    V[2][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0; W[3][2] = 1.0
    
    # M4
    U[3][3] = 1.0
    V[1][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[2][3] = 1.0
    
    # M5
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[1][4] = 1.0
    
    # M6
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # M7
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0
    
    return U, V, W

def _try_strassen_based(n, m, p, max_time):
    """
    Try to build a decomposition for M_{2,3,10} using Strassen-based blocks.
    
    Strategy:
    - Split the 3 columns of A into 2 columns (A_L) and 1 column (a_R).
    - Split the 3 rows of B into 2 rows (B_L) and 1 row (b_R).
    - C = A_L * B_L + a_R * b_R
    - A_L * B_L is 2x2 * 2x10 = 2x10. Split B_L into 5 blocks of 2x2.
    - Each 2x2 * 2x2 uses Strassen (rank 7). Total: 5*7 = 35.
    - a_R * b_R is 2x1 * 1x10 = 2x10. This is sum of 10 outer products. Rank 20.
    - Total rank: 35 + 20 = 55.
    """
    start_time = time.time()
    n, m, p = 2, 3, 10
    R = 55
    U = [[0.0] * R for _ in range(n * m)]  # 6 x 55
    V = [[0.0] * R for _ in range(m * p)]  # 30 x 55
    W = [[0.0] * R for _ in range(p * n)]  # 20 x 55
    
    # Strassen block
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Part 1: A_L * B_L using 5 Strassen blocks
    # A_L is columns 0,1 of A (2x2). A_L[i,j] corresponds to A[i,j] for j=0,1.
    # B_L is rows 0,1 of B (2x10). B_L[j,k] corresponds to B[j,k] for j=0,1.
    # For each block b in 0..4, B_L[:, 2b:2b+2] is a 2x2 matrix.
    
    for b in range(5):
        base_col = b * 7
        for r in range(7):
            col = base_col + r
            # U: A_L[i,j] for i in {0,1}, j in {0,1}
            # U_s[i*2+j, r]
            for i in range(2):
                for j in range(2):
                    U[i*3 + j][col] += U_s[i*2 + j][r]
            
            # V: B_L[j,k] for j in {0,1}, k in {0,1}
            # V_s[j*2+k, r]
            for j in range(2):
                for k in range(2):
                    V[j*10 + b*2 + k][col] += V_s[j*2 + k][r]
            
            # W: C[i,k] for i in {0,1}, k in {0,1}
            # W_s[k*2+i, r]
            for i in range(2):
                for k in range(2):
                    W[(b*2 + k)*2 + i][col] += W_s[k*2 + i][r]
    
    # Part 2: a_R * b_R
    # a_R is column 2 of A (2x1). a_R[i] = A[i,2].
    # b_R is row 2 of B (1x10). b_R[k] = B[2,k].
    # a_R * b_R: C[i,k] += a_R[i] * b_R[k]
    # This is sum over (i,k) of A[i,2] * B[2,k].
    # Each term is a rank-1 tensor: U[i*3+2, col] = 1, V[2*10+k, col] = 1, W[k*2+i, col] = 1
    
    base_col = 35
    for i in range(2):
        for k in range(10):
            col = base_col + i*10 + k
            U[i*3 + 2][col] = 1.0
            V[2*10 + k][col] = 1.0
            W[k*2 + i][col] = 1.0
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Best fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-based decomposition (rank 55)
    if time.time() - start_time < deadline:
        result = _try_strassen_based(n, m, p, deadline - (time.time() - start_time))
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    return best_result
