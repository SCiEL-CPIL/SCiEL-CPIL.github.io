import time

DIMENSIONS = [2, 4, 7]

def _strassen_222():
    """
    Returns the Strassen decomposition for 2x2 by 2x2 matrix multiplication.
    U is 4x7, V is 4x7, W is 4x7.
    
    Row index in U: i*2 + j  (i in 0..1, j in 0..1)
    Row index in V: j*2 + k  (j in 0..1, k in 0..1)
    Row index in W: k*2 + i  (k in 0..1, i in 0..1)
    
    Strassen multipliers:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b12-b22)
    M4 = a22(b21-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    Output:
    c11 = M1 + M4 - M5 + M7
    c12 = M3 + M5
    c21 = M2 + M4
    c22 = M1 - M2 + M3 + M6
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # a11 is row 0, a12 is row 1, a21 is row 2, a22 is row 3
    # b11 is row 0, b12 is row 1, b21 is row 2, b22 is row 3
    # c11 is row 0, c12 is row 1, c21 is row 2, c22 is row 3
    
    # M1: (a11+a22)(b11+b22) -> c11, c22
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # c11
    W[3][0] = 1  # c22
    
    # M2: (a21+a22)b11 -> c21, c22
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # c21
    W[3][1] = -1 # c22
    
    # M3: a11(b12-b22) -> c12, c22
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # c12
    W[3][2] = 1  # c22
    
    # M4: a22(b21-b11) -> c11, c21
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # c11
    W[2][3] = 1  # c21
    
    # M5: (a11+a12)b22 -> c11, c12
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # c11
    W[1][4] = 1  # c12
    
    # M6: (a21-a11)(b11+b12) -> c22
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # c22
    
    # M7: (a12-a22)(b21+b22) -> c11
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # c11
    
    return U, V, W

def _block_221():
    """
    Schoolbook for 2x2 by 2x1 matrix multiplication. Rank 4.
    U is 4x4, V is 2x4, W is 2x4.
    a is 2x2, b is 2x1, c is 2x1.
    c1 = a11*b1 + a12*b2
    c2 = a21*b1 + a22*b2
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # c1: row 0, c2: row 1
    # b1: row 0, b2: row 1
    
    # term 0: a11 * b1 -> c1
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # term 1: a12 * b2 -> c1
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # term 2: a21 * b1 -> c2
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # term 3: a22 * b2 -> c2
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W

def _block_222_schoolbook():
    """
    Schoolbook for 2x2 by 2x2. Rank 4.
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][i*2+j*2+k] = 1
                V[j*2+k][i*2+j*2+k] = 1
                W[k*2+i][i*2+j*2+k] = 1
    return U, V, W

def _block_222():
    return _strassen_222()

def _block_221_schoolbook():
    return _block_221()

def _block_221():
    return _block_221()

def construct():
    deadline = time.time() + 140.0  # safety margin for 150s wall clock
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Decomposition strategy:
    # A (2x4) = [A1, A2] where A1, A2 are 2x2
    # B (4x7) = [B1, B2, B3] where B1 is 4x2, B2 is 4x2, B3 is 4x3
    # Actually, let's split B into 4x2, 4x2, 4x3.
    # C = A1*B1 + A2*B2 + A1*B3_1 + A2*B3_2 where B3 = [B3_1, B3_2] with B3_1 4x1, B3_2 4x2?
    # Wait, B is 4x7. Let's split columns: 2+2+3.
    # B = [B1, B2, B3] where B1 is 4x2, B2 is 4x2, B3 is 4x3.
    # C = A1*B1 + A2*B2 + A1*B3 + A2*B3? No, A is 2x4, so A = [A1, A2].
    # C = [A1, A2] * [B1, B2, B3]
    # C = A1*B1 + A2*B2 + A1*B3 + A2*B3? No.
    # Let's be precise.
    # A = [A1 A2], A1 is 2x2, A2 is 2x2.
    # B = [B1 B2 B3], B1 is 4x2, B2 is 4x2, B3 is 4x3.
    # C = A1*B1 + A2*B2 + A1*B3 + A2*B3? No.
    # C = A*B.
    # Let's split B into B_top (4x2) and B_bottom (4x5)? No.
    
    # Let's use a different splitting.
    # A is 2x4. Split A into A1 (2x2) and A2 (2x2).
    # B is 4x7. Split B into B1 (4x2), B2 (4x2), B3 (4x3).
    # C = A1*B1 + A2*B2 + A1*B3 + A2*B3? No, that's not how matrix multiplication works.
    # C = A1*B1 + A2*B2 only if B is 4x4.
    
    # Let's split B into B1 (4x4) and B2 (4x3).
    # B1 = [B11 B12], B11 4x2, B12 4x2.
    # B2 = [B21 B22], B21 4x1, B22 4x2.
    # C = A1*B11 + A2*B12 + A1*B21 + A2*B22? No.
    # C = A*B = A1*B1 + A2*B2 is wrong.
    # C = A1*B11 + A2*B12 + A1*B21 + A2*B22 is also wrong.
    
    # Correct:
    # A = [A1 A2], B = [B1 B2], B1 is 4x2, B2 is 4x5.
    # C = A1*B1 + A2*B2.
    # A1*B1: A1 is 2x2, B1 is 4x2.
    # Split B1 into B11 (2x2) and B12 (2x2).
    # B1 = [B11; B12].
    # A1*B1 = A1*[B11; B12] = A1_1*B11 + A1_2*B12 where A1 = [A1_1 A1_2]? No, A1 is 2x2.
    # A1*B1 = A1_1*B11 + A1_2*B12 where A1_1 is first row? No.
    # A1 is 2x2. B1 is 4x2.
    # A1 = [[a11 a12], [a21 a22]].
    # B1 = [[b11 b12], [b21 b22], [b31 b32], [b41 b42]].
    # A1*B1 = [[a11*b11 + a12*b21, a11*b12 + a12*b22],
    #          [a21*b11 + a22*b21, a21*b12 + a22*b22]]
    # This is just the top 2 rows of B1 multiplied by A1.
    # So A1*B1 = M(2x2, 2x2) where the 2x2 B matrix is the top 2 rows of B1.
    # But we also have the bottom 2 rows of B1? No, A is 2x4, so it multiplies all 4 rows.
    
    # I was overcomplicating.
    # A is 2x4. B is 4x7.
    # A = [A1 A2], A1 is 2x2, A2 is 2x2.
    # B = [B1; B2], B1 is 2x7, B2 is 2x7.
    # C = A1*B1 + A2*B2.
    # A1*B1: A1 is 2x2, B1 is 2x7.
    # Split B1 into B11 (2x2), B12 (2x2), B13 (2x3).
    # A1*B1 = A1*B11 + A1*B12 + A1*B13.
    # A1*B11: M(2x2, 2x2), rank 7.
    # A1*B12: M(2x2, 2x2), rank 7.
    # A1*B13: M(2x2, 2x3).
    # Split B13 into B131 (2x1), B132 (2x2).
    # A1*B13 = A1*B131 + A1*B132.
    # A1*B131: M(2x2, 2x1), rank 4.
    # A1*B132: M(2x2, 2x2), rank 7.
    # Total for A1*B1: 7+7+4+7 = 25.
    # Similarly for A2*B2: 25.
    # Total rank: 50.
    
    # Let's implement this.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Helper to add a block decomposition
    def add_block(U_blk, V_blk, W_blk, u_offset, v_offset, w_offset):
        """
        U_blk is (n_sub * m_sub) x rank_blk
        V_blk is (m_sub * p_sub) x rank_blk
        W_blk is (p_sub * n_sub) x rank_blk
        """
        rank_blk = U_blk[0].__len__() if U_blk else 0
        n_sub = len(V_blk) // (p_sub := W_blk[0].__len__() if W_blk else 1) # This is messy.
        
    # Let's just manually construct the full U, V, W.
    # Rank = 50.
    rank = 50
    U = [[0]*rank for _ in range(n*m)]  # 2*4 = 8 rows
    V = [[0]*rank for _ in range(m*p)]  # 4*7 = 28 rows
    W = [[0]*rank for _ in range(p*n)]  # 7*2 = 14 rows
    
    col = 0
    
    # A1*B11: A1 is top-left 2x2 of A. B11 is top-left 2x2 of B1 (which is top-left 2x7 of B).
    # A1 rows: 0,1. A1 cols: 0,1.
    # B11 rows: 0,1. B11 cols: 0,1.
    # C11 rows: 0,1. C11 cols: 0,1.
    U11, V11, W11 = _strassen_222()
    for r in range(4):
        U[r][col + r] = U11[r]
    for r in range(4):
        V[r][col + r] = V11[r]
    for r in range(4):
        W[r][col + r] = W11[r]
    col += 7
    
    # A1*B12: A1 is 2x2. B12 is top-right 2x2 of B1 (rows 0,1, cols 2,3).
    U12, V12, W12 = _strassen_222()
    for r in range(4):
        U[r][col + r] = U12[r]
    # V12 rows are for B12. B12 is rows 0,1 of B, cols 2,3.
    # V index: j*p + k. j in 0,1. k in 2,3.
    # So V rows are 2,3,6,7? No.
    # V is 4x7. Row j*p+k.
    # For B12 (rows 0,1, cols 2,3):
    # j=0, k=2 -> row 2
    # j=0, k=3 -> row 3
    # j=1, k=2 -> row 7
    # j=1, k=3 -> row 8
    # Wait, V is m*p = 4*7 = 28 rows.
    # The block V12 is for a 2x2 B matrix.
    # In the local 2x2 B matrix, row j (0,1) and col k (0,1) maps to global row j*p + (offset + k).
    # For B12, offset=2.
    # So local (0,0) -> global (0,2) -> row 0*7+2 = 2.
    # local (0,1) -> global (0,3) -> row 3.
    # local (1,0) -> global (1,2) -> row 7+2 = 9.
    # local (1,1) -> global (1,3) -> row 10.
    for r in range(4):
        local_j, local_k = r // 2, r % 2
        global_row = local_j * p + (2 + local_k)
        V[global_row][col + r] = V12[r]
    # W12 is for C12 (rows 0,1, cols 2,3).
    # Local (k,i) -> global (k, i) with offset.
    # C12 is rows 0,1 (k=0,1) and cols 2,3 (i=2,3).
    # W index: k*n + i.
    # local (0,0) -> global (0,2) -> row 0*2+2 = 2.
    # local (0,1) -> global (0,3) -> row 3.
    # local (1,0) -> global (1,2) -> row 4.
    # local (1,1) -> global (1,3) -> row 5.
    for r in range(4):
        local_k, local_i = r // 2, r % 2
        global_row = local_k * n + (2 + local_i)
        W[global_row][col + r] = W12[r]
    col += 7
    
    # A1*B131: A1 is 2x2. B131 is top-middle 2x1 of B1 (rows 0,1, col 4).
    U131, V131, W131 = _block_221()
    for r in range(4):
        U[r][col + r] = U131[r]
    # V131 rows: j=0,1, k=4.
    # local (0,0) -> global (0,4) -> row 4.
    # local (1,0) -> global (1,4) -> row 11.
    for r in range(2):
        local_j, local_k = r // 2, r % 2
        global_row = local_j * p + (4 + local_k)
        V[global_row][col + r] = V131[r]
    # W131 rows: k=0,1, i=4? No, C131 is rows 0,1, col 4.
    # local (0,0) -> global (0,4) -> row 4.
    # local (1,0) -> global (1,4) -> row 6.
    for r in range(2):
        local_k, local_i = r // 2, r % 2
        global_row = local_k * n + (4 + local_i)
        W[global_row][col + r] = W131[r]
    col += 4
    
    # A1*B132: A1 is 2x2. B132 is top-right 2x2 of B1 (rows 0,1, cols 5,6).
    U132, V132, W132 = _strassen_222()
    for r in range(4):
        U[r][col + r] = U132[r]
    for r in range(4):
        local_j, local_k = r // 2, r % 2
        global_row = local_j * p + (5 + local_k)
        V[global_row][col + r] = V132[r]
    for r in range(4):
        local_k, local_i = r // 2, r % 2
        global_row = local_k * n + (5 + local_i)
        W[global_row][col + r] = W132[r]
    col += 7
    
    # A2*B21: A2 is bottom-right 2x2 of A (rows 2,3). B21 is bottom-left 2x2 of B2 (rows 2,3, cols 0,1).
    U21, V21, W21 = _strassen_222()
    for r in range(4):
        U[2 + r][col + r] = U21[r]
    for r in range(4):
        local_j, local_k = r // 2, r % 2
        global_row = (2 + local_j) * p + local_k
        V[global_row][col + r] = V21[r]
    for r in range(4):
        local_k, local_i = r // 2, r % 2
        global_row = local_k * n + local_i
        W[global_row][col + r] = W21[r]
    col += 7
    
    # A2*B22: A2 is 2x2. B22 is bottom-middle 2x2 of B2 (rows 2,3, cols 2,3).
    U22, V22, W22 = _strassen_222()
    for r in range(4):
        U[2 + r][col + r] = U22[r]
    for r in range(4):
        local_j, local_k = r // 2, r % 2
        global_row = (2 + local_j) * p + (2 + local_k)
        V[global_row][col + r] = V22[r]
    for r in range(4):
        local_k, local_i = r // 2, r % 2
        global_row = local_k * n + (2 + local_i)
        W[global_row][col + r] = W22[r]
    col += 7
    
    # A2*B231: A2 is 2x2. B231 is bottom-middle 2x1 of B2 (rows 2,3, col 4).
    U231, V231, W231 = _block_221()
    for r in range(4):
        U[2 + r][col + r] = U231[r]
    for r in range(2):
        local_j, local_k = r // 2, r % 2
        global_row = (2 + local_j) * p + (4 + local_k)
        V[global_row][col + r] = V231[r]
    for r in range(2):
        local_k, local_i = r // 2, r % 2
        global_row = local_k * n + (4 + local_i)
        W[global_row][col + r] = W231[r]
    col += 4
    
    # A2*B232: A2 is 2x2. B232 is bottom-right 2x2 of B2 (rows 2,3, cols 5,6).
    U232, V232, W232 = _strassen_222()
    for r in range(4):
        U[2 + r][col + r] = U232[r]
    for r in range(4):
        local_j, local_k = r // 2, r % 2
        global_row = (2 + local_j) * p + (5 + local_k)
        V[global_row][col + r] = V232[r]
    for r in range(4):
        local_k, local_i = r // 2, r % 2
        global_row = local_k * n + (5 + local_i)
        W[global_row][col + r] = W232[r]
    col += 7
    
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple test
    import numpy as np
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = DIMENSIONS
    A = np.random.randint(-5, 5, (n, m))
    B = np.random.randint(-5, 5, (m, p))
    C = A @ B
    
    # Verify
    C_check = np.zeros((n, p))
    for r in range(U.shape[1]):
        C_check += np.outer(W[:, r], 1) * np.outer(U[:, r] @ A, B @ V[:, r]) # This is wrong.
    
    # Correct verification:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # C = sum_r U_r A * B V_r * W_r? No.
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] U[j, r]) * (sum_j V[j, r] B[j,k])? No.
    # U is (n*m) x rank. U[i*m+j, r] is the coefficient for A[i,j] in the r-th term.
    # V is (m*p) x rank. V[j*p+k, r] is the coefficient for B[j,k] in the r-th term.
    # W is (p*n) x rank. W[k*n+i, r] is the coefficient for C[i,k] in the r-th term.
    
    # So, term_r = (sum_{i,j} A[i,j] U[i*m+j, r]) * (sum_{j,k} B[j,k] V[j*p+k, r])
    # And C[i,k] = sum_r W[k*n+i, r] * term_r
    
    C_check = np.zeros((n, p))
    for r in range(U.shape[1]):
        term = 0.0
        for i in range(n):
            for j in range(m):
                term += A[i,j] * U[i*m+j, r]
        # This is wrong. The r-th term is a product of a scalar from A and a scalar from B.
        # Let x_r = sum_{i,j} A[i,j] U[i*m+j, r]
        # Let y_r = sum_{j,k} B[j,k] V[j*p+k, r]
        # Then C[i,k] = sum_r W[k*n+i, r] * x_r * y_r? No, that's not bilinear.
        
    # Correct formula:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * u_r[i,j]) * (sum_j v_r[j,k] * B[j,k])
    # Where U is reshaped to (rank, n, m), V to (rank, m, p).
    
    U_r = U.reshape(U.shape[1], n, m)
    V_r = V.reshape(V.shape[1], m, p)
    
    C_check = np.zeros((n, p))
    for r in range(U.shape[1]):
        A_r = A @ U_r[r] # (n, m) @ (m,) -> (n,)
        B_r = V_r[r] @ B # (m,) @ (m, p) -> (p,)
        C_check += W[:, r].reshape(p, n).T * np.outer(A_r, B_r)
    
    print("Max error:", np.max(np.abs(C - C_check)))
    if np.max(np.abs(C - C_check)) < 1e-9:
        print("Validation PASSED")
    else:
        print("Validation FAILED")
