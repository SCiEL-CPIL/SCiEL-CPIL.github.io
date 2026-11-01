DIMENSIONS = [2, 2, 4]
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _trivial_solution(n, m, p):
    """Return the trivial rank n*m*p solution."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_block(A_rows, B_cols, W_rows, n, m, p):
    """
    Build Strassen rank-7 decomposition for one 2x2 * 2x2 block.
    
    A is 2x2 with rows indexed by A_rows (list of (i,j) pairs for U indices i*m+j)
    B is 2x2 with cols indexed by B_cols (list of (j,k) pairs for V indices j*p+k)
    C is 2x2 with W rows indexed by W_rows (list of (k,i) pairs for W indices k*n+i)
    
    Returns: (U_part, V_part, W_part) each of shape (4, 7) for the 4 relevant rows.
    """
    # Strassen products for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map A entries to U rows, B entries to V rows, C entries to W rows.
    # A is 2x2: A[0][0]=A11, A[0][1]=A12, A[1][0]=A21, A[1][1]=A22
    # B is 2x2: B[0][0]=B11, B[0][1]=B12, B[1][0]=B21, B[1][1]=B22
    
    # U rows correspond to A entries: A11, A12, A21, A22 -> U row indices
    # V rows correspond to B entries: B11, B12, B21, B22 -> V row indices
    # W rows correspond to C entries: C11, C12, C21, C22 -> W row indices
    
    # A_rows: list of 4 U row indices for A11, A12, A21, A22
    # B_cols: list of 4 V row indices for B11, B12, B21, B22
    # W_rows: list of 4 W row indices for C11, C12, C21, C22
    
    U_rows = A_rows  # [row_A11, row_A12, row_A21, row_A22]
    V_rows = B_cols  # [row_B11, row_B12, row_B21, row_B22]
    W_rows_local = W_rows  # [row_C11, row_C12, row_C21, row_C22]
    
    rA11, rA12, rA21, rA22 = U_rows
    rB11, rB12, rB21, rB22 = V_rows
    rC11, rC12, rC21, rC22 = W_rows_local
    
    # U matrix: 4 rows x 7 cols
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)*(B11+B22)
    U[rA11, 0] = 1; U[rA22, 0] = 1
    V[rB11, 0] = 1; V[rB22, 0] = 1
    W[rC11, 0] = 1; W[rC22, 0] = 1
    
    # M2 = (A21+A22)*B11
    U[rA21, 1] = 1; U[rA22, 1] = 1
    V[rB11, 1] = 1
    W[rC21, 1] = 1; W[rC22, 1] = -1
    
    # M3 = A11*(B12-B22)
    U[rA11, 2] = 1
    V[rB12, 2] = 1; V[rB22, 2] = -1
    W[rC12, 2] = 1; W[rC22, 2] = 1
    
    # M4 = A22*(B21-B11)
    U[rA22, 3] = 1
    V[rB21, 3] = 1; V[rB11, 3] = -1
    W[rC11, 3] = 1; W[rC21, 3] = 1
    
    # M5 = (A11+A12)*B22
    U[rA11, 4] = 1; U[rA12, 4] = 1
    V[rB22, 4] = 1
    W[rC11, 4] = -1; W[rC12, 4] = 1
    
    # M6 = (A21-A11)*(B11+B12)
    U[rA21, 5] = 1; U[rA11, 5] = -1
    V[rB11, 5] = 1; V[rB12, 5] = 1
    W[rC22, 5] = 1
    
    # M7 = (A12-A22)*(B21+B22)
    U[rA12, 6] = 1; U[rA22, 6] = -1
    V[rB21, 6] = 1; V[rB22, 6] = 1
    W[rC11, 6] = 1
    
    return U, V, W

def _construct_strassen_2x2x4():
    """
    Construct rank-14 decomposition for 2x2 * 2x4 by splitting into two 2x2 * 2x2 blocks.
    
    A is 2x2 (n=2, m=2): rows i=0,1; cols j=0,1
    B is 2x4 (m=2, p=4): rows j=0,1; cols k=0,1,2,3
    C is 2x4 (p=4, n=2): rows k=0,1,2,3; cols i=0,1
    
    Split B into B_left (cols k=0,1) and B_right (cols k=2,3).
    C splits into C_left (rows k=0,1) and C_right (rows k=2,3).
    
    Block 1: A * B_left -> C_left
    Block 2: A * B_right -> C_right
    
    For each block, apply Strassen rank-7.
    
    U has 4 rows (i*m+j): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    V has 8 rows (j*p+k): (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,0)=4, (1,1)=5, (1,2)=6, (1,3)=7
    W has 8 rows (k*n+i): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5, (3,0)=6, (3,1)=7
    """
    n, m, p = 2, 2, 4
    
    # U rows: i*m+j
    # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # V rows: j*p+k
    # (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3
    # (1,0)=4, (1,1)=5, (1,2)=6, (1,3)=7
    
    # W rows: k*n+i
    # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # (2,0)=4, (2,1)=5, (3,0)=6, (3,1)=7
    
    # Block 1: B_left has cols k=0,1
    # B_left[0][0]=B(0,0) -> V row 0, B_left[0][1]=B(0,1) -> V row 1
    # B_left[1][0]=B(1,0) -> V row 4, B_left[1][1]=B(1,1) -> V row 5
    # C_left: rows k=0,1
    # C_left[0][0]=C(0,0) -> W row 0, C_left[0][1]=C(0,1) -> W row 1
    # C_left[1][0]=C(1,0) -> W row 2, C_left[1][1]=C(1,1) -> W row 3
    
    A_rows = [0, 1, 2, 3]  # A11=U[0], A12=U[1], A21=U[2], A22=U[3]
    B_cols_left = [0, 1, 4, 5]  # B11=V[0], B12=V[1], B21=V[4], B22=V[5]
    W_rows_left = [0, 1, 2, 3]  # C11=W[0], C12=W[1], C21=W[2], C22=W[3]
    
    U1, V1, W1 = _strassen_2x2_block(A_rows, B_cols_left, W_rows_left, n, m, p)
    
    # Block 2: B_right has cols k=2,3
    # B_right[0][0]=B(0,2) -> V row 2, B_right[0][1]=B(0,3) -> V row 3
    # B_right[1][0]=B(1,2) -> V row 6, B_right[1][1]=B(1,3) -> V row 7
    # C_right: rows k=2,3
    # C_right[0][0]=C(2,0) -> W row 4, C_right[0][1]=C(2,1) -> W row 5
    # C_right[1][0]=C(3,0) -> W row 6, C_right[1][1]=C(3,1) -> W row 7
    
    B_cols_right = [2, 3, 6, 7]  # B11=V[2], B12=V[3], B21=V[6], B22=V[7]
    W_rows_right = [4, 5, 6, 7]  # C11=W[4], C12=W[5], C21=W[6], C22=W[7]
    
    U2, V2, W2 = _strassen_2x2_block(A_rows, B_cols_right, W_rows_right, n, m, p)
    
    # Now assemble full U (4 x 14), V (8 x 14), W (8 x 14)
    U_full = np.zeros((4, 14))
    V_full = np.zeros((8, 14))
    W_full = np.zeros((8, 14))
    
    # Block 1 occupies columns 0-6
    U_full[:, 0:7] = U1
    V_full[0:4, 0:7] = V1[0:4, :]  # V rows for B_left are V[0],V[1],V[4],V[5]
    # Wait, V1 is indexed by local rows 0-3 corresponding to B11,B12,B21,B22
    # We need to map V1's local rows to global V rows
    # V1 row 0 -> V_full row 0 (B11)
    # V1 row 1 -> V_full row 1 (B12)
    # V1 row 2 -> V_full row 4 (B21)
    # V1 row 3 -> V_full row 5 (B22)
    v_map_left = [0, 1, 4, 5]
    for local_r, global_r in enumerate(v_map_left):
        V_full[global_r, 0:7] = V1[local_r, :]
    
    # W1 row 0 -> W_full row 0 (C11)
    # W1 row 1 -> W_full row 1 (C12)
    # W1 row 2 -> W_full row 2 (C21)
    # W1 row 3 -> W_full row 3 (C22)
    w_map_left = [0, 1, 2, 3]
    for local_r, global_r in enumerate(w_map_left):
        W_full[global_r, 0:7] = W1[local_r, :]
    
    # Block 2 occupies columns 7-13
    U_full[:, 7:14] = U2
    # V2 row 0 -> V_full row 2 (B11 of right block)
    # V2 row 1 -> V_full row 3 (B12 of right block)
    # V2 row 2 -> V_full row 6 (B21 of right block)
    # V2 row 3 -> V_full row 7 (B22 of right block)
    v_map_right = [2, 3, 6, 7]
    for local_r, global_r in enumerate(v_map_right):
        V_full[global_r, 7:14] = V2[local_r, :]
    
    # W2 row 0 -> W_full row 4 (C11 of right block)
    # W2 row 1 -> W_full row 5 (C12 of right block)
    # W2 row 2 -> W_full row 6 (C21 of right block)
    # W2 row 3 -> W_full row 7 (C22 of right block)
    w_map_right = [4, 5, 6, 7]
    for local_r, global_r in enumerate(w_map_right):
        W_full[global_r, 7:14] = W2[local_r, :]
    
    return U_full.tolist(), V_full.tolist(), W_full.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deterministic Strassen-based rank-14 construction
    U, V, W = _construct_strassen_2x2x4()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to trivial solution (should not be reached)
    return _trivial_solution(n, m, p)
