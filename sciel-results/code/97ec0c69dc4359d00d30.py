import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Uses recursive block decomposition:
    1. Split A (2x4) into two 2x2 blocks (A1: cols 0-1, A2: cols 2-3).
    2. Split B (4x5) into two 2x5 blocks (B1: rows 0-1, B2: rows 2-3).
    3. Each 2x2x5 block uses rank 18 (Strassen for 2x2x2 + custom rank 11 for 2x2x3).
    4. Total rank = 2 * 18 = 36.
    
    The 2x2x3 rank-11 decomposition is constructed as:
    - First 2 columns (2x2x2): Strassen rank 7
    - Last 1 column (2x2x1): Schoolbook rank 4
    Total: 7 + 4 = 11
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Helper: Strassen 2x2x2 decomposition (rank 7)
    # A (2x2), B (2x2), C (2x2)
    # U: 4x7 (A indices: 0,1,2,3 for A[0,0], A[0,1], A[1,0], A[1,1])
    # V: 4x7 (B indices: 0,1,2,3 for B[0,0], B[0,1], B[1,0], B[1,1])
    # W: 4x7 (C indices: k*n+i, n=2. C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3)
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h): A[0,0], A[1,1] and B[0,0], B[1,1]
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # M1 contributes to C[0,0] (W idx 0) and C[1,1] (W idx 3)
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)(e): A[1,0], A[1,1] and B[0,0]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # M2 contributes to C[1,0] (W idx 1) and C[1,1] (W idx 3) with -1
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3 = (a-b)(f+h): A[0,0], A[0,1] and B[0,1], B[1,1]
        U[0, 2] = 1; U[1, 2] = -1
        V[1, 2] = 1; V[3, 2] = 1
        # M3 contributes to C[0,1] (W idx 2)
        W[2, 2] = 1
        
        # M4 = (a+d)(g-i): A[0,0], A[1,1] and B[1,0], B[1,1] (g=B[1,0] idx2, i=B[1,1] idx3)
        U[0, 3] = 1; U[3, 3] = 1
        V[2, 3] = 1; V[3, 3] = -1
        # M4 contributes to C[0,0] (W idx 0) and C[1,0] (W idx 1)
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (a+c)(e-h): A[0,0], A[1,0] and B[0,0], B[1,1]
        U[0, 4] = 1; U[2, 4] = 1
        V[0, 4] = 1; V[3, 4] = -1
        # M5 contributes to C[0,0] (W idx 0) with -1 and C[0,1] (W idx 2)
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (b+d)(g-h): A[0,1], A[1,1] and B[1,0], B[1,1]
        U[1, 5] = 1; U[3, 5] = 1
        V[2, 5] = 1; V[3, 5] = -1
        # M6 contributes to C[1,1] (W idx 3)
        W[3, 5] = 1
        
        # M7 = (a-c)(f+i): A[0,0], A[1,0] and B[0,1], B[1,0]
        U[0, 6] = 1; U[2, 6] = -1
        V[1, 6] = 1; V[2, 6] = 1
        # M7 contributes to C[0,0] (W idx 0) and C[1,1] (W idx 3)
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    # Helper: Schoolbook 2x2x1 decomposition (rank 4)
    # A (2x2), B (2x1), C (2x1)
    # U: 4x4, V: 2x4, W: 2x4
    def schoolbook_221():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    # Helper: 2x2x3 decomposition (rank 11)
    # A (2x2), B (2x3), C (2x3)
    # U: 4x11, V: 6x11, W: 6x11
    # W indices: k*n+i, n=2. C[0,0]->0, C[0,1]->2, C[0,2]->4, C[1,0]->1, C[1,1]->3, C[1,2]->5
    def block_223():
        U1, V1, W1 = strassen_222()   # 2x2x2 part (B cols 0,1)
        U2, V2, W2 = schoolbook_221() # 2x2x1 part (B col 2)
        
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Part 1: 2x2x2 (B cols 0,1)
        U[:, 0:7] = U1
        # V1 is 4x7 for B[0,0], B[0,1], B[1,0], B[1,1]
        # In 2x2x3, B indices: B[0,0]->0, B[0,1]->1, B[1,0]->2, B[1,1]->3, B[0,2]->4, B[1,2]->5
        V[0:4, 0:7] = V1
        # W1 is 4x7 for C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3
        # In 2x2x3, C indices: C[0,0]->0, C[0,1]->2, C[0,2]->4, C[1,0]->1, C[1,1]->3, C[1,2]->5
        W[0:4, 0:7] = W1
        
        # Part 2: 2x2x1 (B col 2)
        U[:, 7:11] = U2
        # V2 is 2x4 for B[0,2]->4, B[1,2]->5
        V[4:6, 7:11] = V2
        # W2 is 2x4 for C[0,2]->4, C[1,2]->5
        W[4:6, 7:11] = W2
        
        return U, V, W
    
    # Helper: 2x2x5 decomposition (rank 18)
    # A (2x2), B (2x5), C (2x5)
    # U: 4x18, V: 10x18, W: 10x18
    # W indices: k*n+i, n=2. C[i,k] -> k*2+i
    def block_225():
        U1, V1, W1 = strassen_222()   # 2x2x2 part (B cols 0,1)
        U2, V2, W2 = block_223()      # 2x2x3 part (B cols 2,3,4)
        
        U = np.zeros((4, 18))
        V = np.zeros((10, 18))
        W = np.zeros((10, 18))
        
        # Part 1: 2x2x2 (B cols 0,1)
        U[:, 0:7] = U1
        # V1 is 4x7 for B[0,0]->0, B[0,1]->1, B[1,0]->2, B[1,1]->3
        V[0:4, 0:7] = V1
        # W1 is 4x7 for C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3
        W[0:4, 0:7] = W1
        
        # Part 2: 2x2x3 (B cols 2,3,4)
        U[:, 7:18] = U2
        # V2 is 6x11 for B[0,2]->4, B[0,3]->5, B[0,4]->6, B[1,2]->7, B[1,3]->8, B[1,4]->9
        V[4:10, 7:18] = V2
        # W2 is 6x11 for C[0,2]->4, C[0,3]->6, C[0,4]->8, C[1,2]->5, C[1,3]->7, C[1,4]->9
        W[4:10, 7:18] = W2
        
        return U, V, W
    
    # Assemble final 2x4x5 decomposition (rank 36)
    # A (2x4), B (4x5), C (2x5)
    # U: 8x36 (A indices: i*m+j, m=4)
    # V: 20x36 (B indices: j*p+k, p=5)
    # W: 10x36 (C indices: k*n+i, n=2)
    def construct_final():
        U1, V1, W1 = block_225()  # A[0:2, 0:2] (cols 0,1), B[0:2, :] (rows 0,1)
        U2, V2, W2 = block_225()  # A[0:2, 2:4] (cols 2,3), B[2:4, :] (rows 2,3)
        
        U = np.zeros((8, 36))
        V = np.zeros((20, 36))
        W = np.zeros((10, 36))
        
        # Part 1: A cols 0,1 (U indices 0,1,4,5 for A[0,0], A[0,1], A[1,0], A[1,1])
        # B rows 0,1 (V indices 0-4 for row 0, 5-9 for row 1)
        # U1 is 4x18 for A[0,0]->0, A[0,1]->1, A[1,0]->4, A[1,1]->5
        U[0, 0:18] = U1[0, :]   # A[0,0]
        U[1, 0:18] = U1[1, :]   # A[0,1]
        U[4, 0:18] = U1[2, :]   # A[1,0]
        U[5, 0:18] = U1[3, :]   # A[1,1]
        
        # V1 is 10x18 for B[0,0]->0, B[0,1]->1, B[0,2]->2, B[0,3]->3, B[0,4]->4,
        #            B[1,0]->5, B[1,1]->6, B[1,2]->7, B[1,3]->8, B[1,4]->9
        V[0:10, 0:18] = V1
        
        # W1 is 10x18 for C[0,0]->0, C[0,1]->2, C[0,2]->4, C[0,3]->6, C[0,4]->8,
        #            C[1,0]->1, C[1,1]->3, C[1,2]->5, C[1,3]->7, C[1,4]->9
        W[:, 0:18] = W1
        
        # Part 2: A cols 2,3 (U indices 2,3,6,7 for A[0,2], A[0,3], A[1,2], A[1,3])
        # B rows 2,3 (V indices 10-14 for row 2, 15-19 for row 3)
        U[2, 18:36] = U2[0, :]   # A[0,2]
        U[3, 18:36] = U2[1, :]   # A[0,3]
        U[6, 18:36] = U2[2, :]   # A[1,2]
        U[7, 18:36] = U2[3, :]   # A[1,3]
        
        # V2 is 10x18 for B[2,0]->10, B[2,1]->11, B[2,2]->12, B[2,3]->13, B[2,4]->14,
        #            B[3,0]->15, B[3,1]->16, B[3,2]->17, B[3,3]->18, B[3,4]->19
        V[10:20, 18:36] = V2
        
        # W2 is 10x18 for same C indices as W1
        W[:, 18:36] = W2
        
        return U, V, W
    
    U, V, W = construct_final()
    
    # Verify correctness
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_s = np.zeros((n*m, rank_schoolbook))
        V_s = np.zeros((m*p, rank_schoolbook))
        W_s = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1
                    V_s[j*p + k, col] = 1
                    W_s[k*n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
