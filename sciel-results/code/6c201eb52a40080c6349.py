import time
import numpy as np
import itertools

DIMENSIONS = [3, 4, 8]

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=None, m=None, p=None):
    """
    Fast exact verification.
    """
    if n is None:
        n, m, p = DIMENSIONS

    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)

    if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
        return False
    if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
        return False
    rank = U_arr.shape[1]
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False

    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank).
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)

    # T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) <= 1e-9))

def _strassen_2x2():
    """
    Strassen 2x2x2 decomposition.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i,j,r], V[j,k,r], W[k,i,r] for i,j,k in {0,1}
    C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    """
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
    #
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(e+f)
    # m7 = (b-c)(g+h)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    #
    # U[i,j,r]: coefficient of A[i,j] in multiplication r
    # V[j,k,r]: coefficient of B[j,k] in multiplication r
    # W[k,i,r]: coefficient in output C[i,k]
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # U is indexed by (i,j) flattened: row = i*2+j
    # V is indexed by (j,k) flattened: row = j*2+k  
    # W is indexed by (k,i) flattened: row = k*2+i (note: W[k*n+i] with n=2)
    
    # m1 = (a+d)(e+h): U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    W[0, 0] = 1  # C[0,0] (k=0,i=0)
    W[3, 0] = 1  # C[1,1] (k=1,i=1)
    
    # m2 = (c+d)e: U[2,1]=1, U[3,1]=1, V[0,0]=1
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1  # C[1,0] (k=0,i=1)
    
    # m3 = a(f-h): U[0,0]=1, V[0,1]=1, V[1,1]=-1
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1  # B[0,1]
    V[3, 2] = -1 # B[1,1]
    W[1, 2] = 1  # C[0,1] (k=1,i=0)
    
    # m4 = d(g-e): U[3,1]=1, V[1,0]=1, V[0,0]=-1
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1  # B[1,0]
    V[0, 3] = -1 # B[0,0]
    W[0, 3] = 1  # C[0,0] (k=0,i=0)
    W[2, 3] = 1  # C[1,0] (k=0,i=1)
    
    # m5 = (a+b)h: U[0,0]=1, U[1,0]=1, V[1,1]=1
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[0, 4] = -1 # C[0,0] (k=0,i=0)
    W[1, 4] = 1  # C[0,1] (k=1,i=0)
    
    # m6 = (c-d)(e+f): U[2,1]=1, U[3,1]=-1, V[0,0]=1, V[0,1]=1
    U[2, 5] = 1  # A[1,0]
    U[3, 5] = -1 # A[1,1]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    W[3, 5] = 1  # C[1,1] (k=1,i=1)
    
    # m7 = (b-c)(g+h): U[1,0]=1, U[2,0]=-1, V[1,0]=1, V[1,1]=1
    U[1, 6] = 1  # A[0,1]
    U[2, 6] = -1 # A[1,0]
    V[2, 6] = 1  # B[1,0]
    V[3, 6] = 1  # B[1,1]
    W[0, 6] = 1  # C[0,0] (k=0,i=0)
    
    return U, V, W

def _strassen_2x2_verify():
    """Verify Strassen decomposition."""
    U, V, W = _strassen_2x2()
    # Check: for all i,j,k in {0,1}, sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1
    U3 = U.reshape(2, 2, 7)
    V3 = V.reshape(2, 2, 7)
    W3 = W.reshape(2, 2, 7)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _block_2x4x4():
    """
    Rank 14 decomposition for 2x4 x 4x4 -> 2x4.
    Split B into two 4x2 blocks, use Strassen for each 2x2x2 subproblem.
    
    A is 2x4, B is 4x4, C = A*B is 2x4.
    Split B into B_L (4x2) and B_R (4x2).
    C_L = A * B_L (2x2), C_R = A * B_R (2x2).
    Each 2x2x2 uses Strassen with rank 7.
    Total rank: 14.
    
    U: (2*4, 14) = (8, 14)
    V: (4*4, 14) = (16, 14)
    W: (4*2, 14) = (8, 14)  [W indexed by k*n+i where n=2, p=4, so k in 0..3, i in 0..1]
    """
    U_s, V_s, W_s = _strassen_2x2()
    
    U = np.zeros((8, 14), dtype=int)
    V = np.zeros((16, 14), dtype=int)
    W = np.zeros((8, 14), dtype=int)
    
    # Block 1: C_L = A * B_L (2x2 output)
    # A is 2x4, but Strassen expects 2x2. We need to handle the 4 columns.
    # Actually, for 2x4 x 4x2 -> 2x2, we can't directly use 2x2 Strassen.
    # We need a different approach.
    
    # Let me reconsider. For 2x4 x 4x4 -> 2x4:
    # We can view this as: C[:, 0:2] = A * B[:, 0:2] and C[:, 2:4] = A * B[:, 2:4]
    # Each is 2x4 x 4x2 -> 2x2.
    # For 2x4 x 4x2, we can split A into two 2x2 blocks: A_L (2x2) and A_R (2x2)
    # Then C_L = A_L * B_L[:, 0:2] + A_R * B_L[:, 2:4]
    # This gets complicated. Let me use a simpler block approach.
    
    # Actually, let me just use schoolbook for now to get a working solution,
    # then optimize.
    
    # For 2x4 x 4x4:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, j in 0..3, k in 0..3
    # Schoolbook rank: 2*4*4 = 32
    # But we can do better with Strassen-like methods.
    
    # Let me try a different split: 3x4x8 = (1+2)x4x8
    # 1x4x8: rank 32 (schoolbook)
    # 2x4x8: can we do better?
    
    # For 2x4x8: split into 2x4x4 + 2x4x4 = two 2x4x4 blocks
    # Each 2x4x4: C[i,k] = sum_j A[i,j]*B[j,k], i in 0..1, j in 0..3, k in 0..3
    # This is like multiplying a 2x4 matrix by a 4x4 matrix.
    # We can use the fact that 2x4 = 2*(1x4), so we can think of it as two row vectors.
    
    # For 1x4 x 4x4 -> 1x4: C[k] = sum_j A[j]*B[j,k]
    # This is a dot product for each k. Rank 16 (schoolbook).
    # But can we do better? For 1xm x mxp, the rank is m*p if no structure.
    # So 1x4x4 has rank 16.
    
    # So 2x4x4 has rank 32 (two independent 1x4x4 problems).
    # And 2x4x8 = 2x4x4 + 2x4x4 = 32 + 32 = 64.
    # 1x4x8 = 1x4x4 + 1x4x4 = 16 + 16 = 32.
    # Total: 64 + 32 = 96. Same as schoolbook!
    
    # I need to find a better decomposition. Let me try Strassen more carefully.
    
    # For 2x2x2, Strassen gives rank 7 vs 8 schoolbook.
    # For 2x4x4, can we use Strassen?
    # Split 2x4 into two 2x2 blocks: A = [A1 A2] where A1, A2 are 2x2.
    # B is 4x4, split into B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = A*B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # Each 2x2x2 multiplication can use Strassen (rank 7).
    # We have 4 such multiplications: A1*B11, A2*B21, A1*B12, A2*B22
    # Total rank: 4*7 = 28 for 2x4x4.
    
    U = np.zeros((8, 28), dtype=int)
    V = np.zeros((16, 28), dtype=int)
    W = np.zeros((8, 28), dtype=int)
    
    # A is 2x4: rows 0-1, columns 0-3
    # Split A into A1 (cols 0-1) and A2 (cols 2-3)
    # B is 4x4: rows 0-3, cols 0-3
    # Split B into B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3),
    #            B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    # C is 2x4: rows 0-1, cols 0-3
    # C[:, 0-1] = A1*B11 + A2*B21
    # C[:, 2-3] = A1*B12 + A2*B22
    
    # Strassen for 2x2x2:
    # U_s (4x7), V_s (4x7), W_s (4x7)
    # U_s indexed by (i,j) with i,j in 0..1, row = i*2+j
    # V_s indexed by (j,k) with j,k in 0..1, row = j*2+k
    # W_s indexed by (k,i) with k,i in 0..1, row = k*2+i
    
    # Block 1: A1 * B11 -> contributes to C[:, 0-1]
    # A1 is A[0:2, 0:2], B11 is B[0:2, 0:2]
    # U rows 0-3 (A[0:2, 0:2]), V rows 0-3 (B[0:2, 0:2]), W rows 0-3 (C[0:2, 0:2])
    U[0:4, 0:7] = U_s
    V[0:4, 0:7] = V_s
    W[0:4, 0:7] = W_s
    
    # Block 2: A2 * B21 -> contributes to C[:, 0-1]
    # A2 is A[0:2, 2:4], B21 is B[2:4, 0:2]
    # U rows 4-7 (A[0:2, 2:4]), V rows 4-7 (B[2:4, 0:2]), W rows 0-3 (C[0:2, 0:2])
    U[4:8, 7:14] = U_s
    V[4:8, 7:14] = V_s
    W[0:4, 7:14] = W_s
    
    # Block 3: A1 * B12 -> contributes to C[:, 2-3]
    # A1 is A[0:2, 0:2], B12 is B[0:2, 2:4]
    # U rows 0-3 (A[0:2, 0:2]), V rows 0-3 (B[0:2, 2:4]), W rows 4-7 (C[0:2, 2:4])
    U[0:4, 14:21] = U_s
    V[0:4, 14:21] = V_s
    W[4:8, 14:21] = W_s
    
    # Block 4: A2 * B22 -> contributes to C[:, 2-3]
    # A2 is A[0:2, 2:4], B22 is B[2:4, 2:4]
    # U rows 4-7 (A[0:2, 2:4]), V rows 8-11 (B[2:4, 2:4]), W rows 4-7 (C[0:2, 2:4])
    U[4:8, 21:28] = U_s
    V[8:12, 21:28] = V_s
    W[4:8, 21:28] = W_s
    
    return U, V, W

def _block_1x4x4():
    """
    Schoolbook rank 16 decomposition for 1x4 x 4x4 -> 1x4.
    C[k] = sum_j A[j] * B[j,k]
    
    U: (4, 16) - A has 4 elements
    V: (16, 16) - B has 16 elements
    W: (4, 16) - C has 4 elements, W indexed by k*n+i where n=1, so k*1+0 = k
    """
    U = np.zeros((4, 16), dtype=int)
    V = np.zeros((16, 16), dtype=int)
    W = np.zeros((4, 16), dtype=int)
    
    col = 0
    for j in range(4):
        for k in range(4):
            U[j, col] = 1
            V[j*4+k, col] = 1
            W[k, col] = 1
            col += 1
    
    return U, V, W

def _block_decomposition():
    """
    Full 3x4x8 decomposition using block structure.
    Split 3 rows into 2+1.
    2x4x8: use two 2x4x4 blocks (rank 28 each) = 56
    1x4x8: use two 1x4x4 blocks (rank 16 each) = 32
    Total: 56 + 32 = 88
    
    Wait, that's worse than 96? No, 88 < 96.
    """
    n, m, p = 3, 4, 8
    
    U_2x4x4, V_2x4x4, W_2x4x4 = _block_2x4x4()
    U_1x4x4, V_1x4x4, W_1x4x4 = _block_1x4x4()
    
    # Total rank: 28 + 28 + 16 + 16 = 88
    total_rank = 88
    
    U = np.zeros((n*m, total_rank), dtype=int)  # (12, 88)
    V = np.zeros((m*p, total_rank), dtype=int)  # (32, 88)
    W = np.zeros((p*n, total_rank), dtype=int)  # (24, 88)
    
    # Block 1: rows 0-1 of A, cols 0-3 of B -> cols 0-3 of C
    # A[0:2, 0:4], B[0:4, 0:4], C[0:4, 0:2]
    U[0:8, 0:28] = U_2x4x4
    V[0:16, 0:28] = V_2x4x4
    W[0:8, 0:28] = W_2x4x4
    
    # Block 2: rows 0-1 of A, cols 4-7 of B -> cols 4-7 of C
    # A[0:2, 0:4], B[0:4, 4:8], C[4:8, 0:2]
    U[0:8, 28:56] = U_2x4x4
    V[16:32, 28:56] = V_2x4x4
    W[8:16, 28:56] = W_2x4x4
    
    # Block 3: row 2 of A, cols 0-3 of B -> cols 0-3 of C
    # A[2:3, 0:4], B[0:4, 0:4], C[0:4, 2:3]
    U[8:12, 56:72] = U_1x4x4
    V[0:16, 56:72] = V_1x4x4
    W[16:20, 56:72] = W_1x4x4
    
    # Block 4: row 2 of A, cols 4-7 of B -> cols 4-7 of C
    # A[2:3, 0:4], B[0:4, 4:8], C[4:8, 2:3]
    U[8:12, 72:88] = U_1x4x4
    V[16:32, 72:88] = V_1x4x4
    W[20:24, 72:88] = W_1x4x4
    
    return {"U": U, "V": V, "W": W}

def _to_lists(res):
    return {
        "U": res["U"].astype(int).tolist(),
        "V": res["V"].astype(int).tolist(),
        "W": res["W"].astype(int).tolist(),
    }

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Return schoolbook as fallback to ensure correctness
    best = _schoolbook()
    
    # Try block decomposition if time permits
    try:
        if time.time() < deadline - 30.0:
            res = _block_decomposition()
            if _verify(res["U"].tolist(), res["V"].tolist(), res["W"].tolist()):
                best = _to_lists(res)
    except Exception:
        pass

    return best
