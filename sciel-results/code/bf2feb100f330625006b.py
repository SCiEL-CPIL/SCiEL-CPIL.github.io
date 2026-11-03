import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _strassen_222():
    """Strassen decomposition for 2x2 * 2x2.
    U shape (4, 7): U[i*2+j, r]
    V shape (4, 7): V[j*2+k, r]
    W shape (4, 7): W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00+a11)(b00+b11)
    U[0, 0] += 1.0; U[3, 0] += 1.0
    V[0, 0] += 1.0; V[3, 0] += 1.0
    W[0, 0] += 1.0; W[3, 0] += 1.0
    # M2: (a10+a11)b00
    U[2, 1] += 1.0; U[3, 1] += 1.0
    V[0, 1] += 1.0
    W[2, 1] += 1.0; W[3, 1] += -1.0
    # M3: a00(b01-b11)
    U[0, 2] += 1.0
    V[1, 2] += 1.0; V[3, 2] += -1.0
    W[1, 2] += 1.0; W[3, 2] += 1.0
    # M4: a11(b10-b00)
    U[3, 3] += 1.0
    V[2, 3] += 1.0; V[0, 3] += -1.0
    W[0, 3] += 1.0; W[2, 3] += 1.0
    # M5: (a00+a01)b11
    U[0, 4] += 1.0; U[1, 4] += 1.0
    V[3, 4] += 1.0
    W[0, 4] += -1.0; W[1, 4] += 1.0
    # M6: (a10-a00)(b00+b01)
    U[2, 5] += 1.0; U[0, 5] += -1.0
    V[0, 5] += 1.0; V[1, 5] += 1.0
    W[3, 5] += 1.0
    # M7: (a00+a01)(b10+b11)
    U[0, 6] += 1.0; U[1, 6] += 1.0
    V[2, 6] += 1.0; V[3, 6] += 1.0
    W[0, 6] += 1.0
    
    return U, V, W

def _strassen_223():
    """Decompose 2x2 * 2x3.
    A: 2x2, B: 2x3, C: 3x2
    U shape (4, rank): U[i*2+j, r]
    V shape (6, rank): V[j*3+k, r]
    W shape (6, rank): W[k*2+i, r]
    
    Split B into B1 (2x2) and B2 (2x1).
    C = A*B1 + A*B2
    A*B1: 2x2*2x2, rank 7
    A*B2: 2x2*2x1, rank 4
    Total rank: 11
    """
    U_s, V_s, W_s = _strassen_222()
    
    # 2x2 * 2x1: A(2x2), B(2x1), C(1x2)
    # U rows: i*2+j, i,j in 0-1 -> 4 rows
    # V rows: j*1+k = j, j in 0-1 -> 2 rows
    # W rows: k*2+i = i, i in 0-1 -> 2 rows
    # C[0,i] = sum_j A[i,j]*B[j,0]
    U_221 = np.zeros((4, 4))
    V_221 = np.zeros((2, 4))
    W_221 = np.zeros((2, 4))
    for j in range(2):
        for i in range(2):
            U_221[i*2+j, j] = 1.0
            V_221[j, j] = 1.0
            W_221[i, j] = 1.0
    
    rank = 7 + 4
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    # First part: A * B1 (2x2 * 2x2), B1 is rows 0-1 of B (j=0,1), cols 0-1 (k=0,1)
    # V rows for B1: j*3+k, j in 0-1, k in 0-1 -> rows 0,1,3,4
    for r in range(7):
        U[:, r] = U_s[:, r]
        # Map V: j*2+k -> j*3+k
        for j in range(2):
            for k in range(2):
                V[j*3+k, r] = V_s[j*2+k, r]
        # Map W: k*2+i -> k*2+i (same for k in 0-1)
        for k in range(2):
            for i in range(2):
                W[k*2+i, r] = W_s[k*2+i, r]
    
    # Second part: A * B2 (2x2 * 2x1), B2 is rows 0-1 of B (j=0,1), col 2 (k=2)
    # V rows for B2: j*3+2, j in 0-1 -> rows 2, 5
    for r in range(4):
        col = 7 + r
        U[:, col] = U_221[:, r]
        for j in range(2):
            V[j*3+2, col] = V_221[j, r]
        for i in range(2):
            W[i, col] = W_221[i, r]
    
    return U, V, W

def _decompose_347():
    """Decompose 3x4 * 4x7.
    Split 3 = 2+1, 4 = 2+2, 7 = 2+2+3
    (2x4)*(4x7) + (1x4)*(4x7)
    
    2x4*4x7: split 4=2+2, 7=2+2+3
    For each pair (A_block 2x2, B_block 2xp):
      2x2*2x2: rank 7 (Strassen)
      2x2*2x3: rank 11
      
    2x4*4x2: 2x2*2x2 + 2x2*2x2 = 7+7 = 14
    2x4*4x2: 14
    2x4*4x3: 2x2*2x3 + 2x2*2x3 = 11+11 = 22
    Total 2x4*4x7: 14+14+22 = 50
    
    1x4*4x7: rank 4 (each of 4 products)
    Total: 50+4 = 54
    """
    # Build 2x2*2x2 Strassen blocks
    S = _strassen_222()
    S223 = _strassen_223()
    
    # 2x4*4x2 (first 2 cols of B)
    # A_left (2x2) * B_top (2x2): Strassen
    # A_right (2x2) * B_top (2x2): Strassen
    # A_left (2x2) * B_bot (2x2): Strassen  
    # A_right (2x2) * B_bot (2x2): Strassen
    # Wait, 2x4*4x2: A is 2x4, B is 4x2
    # A = [A11 A12] where A11, A12 are 2x2
    # B = [B11; B21] where B11, B21 are 2x2
    # C = A11*B11 + A12*B21
    # Each 2x2*2x2 is rank 7, total 14
    
    # 2x4*4x3: A is 2x4, B is 4x3
    # A = [A11 A12], B = [B1; B2] where B1 is 2x3, B2 is 2x3
    # C = A11*B1 + A12*B2
    # Each 2x2*2x3 is rank 11, total 22
    
    # Total 2x4*4x7: 14 + 14 + 22 = 50
    # 1x4*4x7: rank 4
    # Grand total: 54
    
    rank_total = 54
    U = np.zeros((12, rank_total))  # 3*4 = 12
    V = np.zeros((28, rank_total))  # 4*7 = 28
    W = np.zeros((21, rank_total))  # 7*3 = 21
    
    col = 0
    # 2x4*4x2: 4 Strassen 2x2*2x2 blocks
    # A rows 0-3 (2x2 left), 4-7 (2x2 right)
    # B rows 0-3 (top 2x2), 4-7 (bottom 2x2)
    # C: 2x2 output, W rows k*3+i, k in 0-1, i in 0-1 -> rows 0,1,3,4
    
    # Block (0,0): A11 * B11
    for r in range(7):
        U[0:4, col+r] = S[0:4, r]
        V[0:4, col+r] = S[0:4, r]
        # W: k*3+i for k in 0-1, i in 0-1
        for k in range(2):
            for i in range(2):
                W[k*3+i, col+r] = S[k*2+i, r]
    col += 7
    
    # Block (0,1): A12 * B21
    for r in range(7):
        U[4:8, col+r] = S[0:4, r]
        V[4:8, col+r] = S[0:4, r]
        for k in range(2):
            for i in range(2):
                W[k*3+i, col+r] = S[k*2+i, r]
    col += 7
    
    # 2x4*4x2 (second 2 cols of B, cols 2-3)
    # B rows 8-11 (top 2x2 for col 2-3), 12-15 (bottom 2x2)
    # Wait, B is 4x7. V index is j*7+k.
    # B rows: j in 0-3, k in 0-6
    # For first 2 cols (k=0,1): V rows j*7+0, j*7+1 for j in 0-3
    # For cols 2-3: V rows j*7+2, j*7+3
    # For cols 4-5: V rows j*7+4, j*7+5
    # For col 6: V row j*7+6
    
    # Let me redo with proper indexing.
    # V[j*7+k, r] for j in 0-3, k in 0-6
    
    # Actually this is getting complicated. Let me use a cleaner approach.
    # I'll build it block by block with explicit index tracking.
    
    U = np.zeros((12, 54))
    V = np.zeros((28, 54))
    W = np.zeros((21, 54))
    
    col = 0
    # 2x4*4x7 part (A rows 0-1, i.e., U rows 0-7)
    # Split A into A1 (cols 0-1, U rows 0-3) and A2 (cols 2-3, U rows 4-7)
    # Split B into B1 (cols 0-1), B2 (cols 2-3), B3 (cols 4-6)
    # B1: V rows j*7+0, j*7+1 for j in 0-3 -> rows 0,1,7,8
    # B2: V rows j*7+2, j*7+3 for j in 0-3 -> rows 2,3,9,10
    # B3: V rows j*7+4, j*7+5, j*7+6 for j in 0-3 -> rows 4,5,6,11,12,13,18,19,20
    
    # A1*B1: 2x2*2x2, Strassen rank 7
    # U rows: 0-3, V rows: 0,1,7,8, W rows: k*3+i for k in 0-1, i in 0-1
    S = _strassen_222()
    v_rows_B1 = [0, 1, 7, 8]  # j*7+k for j in 0-1, k in 0-1
    for r in range(7):
        U[0:4, col+r] = S[0:4, r]
        for idx, vr in enumerate(v_rows_B1):
            V[vr, col+r] = S[idx, r]
        for k in range(2):
            for i in range(2):
                W[k*3+i, col+r] = S[k*2+i, r]
    col += 7
    
    # A2*B1: 2x2*2x2, Strassen rank 7
    v_rows_B1 = [0, 1, 7, 8]
    for r in range(7):
        U[4:8, col+r] = S[0:4, r]
        for idx, vr in enumerate(v_rows_B1):
            V[vr, col+r] = S[idx, r]
        for k in range(2):
            for i in range(2):
                W[k*3+i, col+r] = S[k*2+i, r]
    col += 7
    
    # A1*B2: 2x2*2x2, Strassen rank 7
    v_rows_B2 = [2, 3, 9, 10]
    for r in range(7):
        U[0:4, col+r] = S[0:4, r]
        for idx, vr in enumerate(v_rows_B2):
            V[vr, col+r] = S[idx, r]
        for k in range(2):
            for i in range(2):
                W[(k+2)*3+i, col+r] = S[k*2+i, r]
    col += 7
    
    # A2*B2: 2x2*2x2, Strassen rank 7
    v_rows_B2 = [2, 3, 9, 10]
    for r in range(7):
        U[4:8, col+r] = S[0:4, r]
        for idx, vr in enumerate(v_rows_B2):
            V[vr, col+r] = S[idx, r]
        for k in range(2):
            for i in range(2):
                W[(k+2)*3+i, col+r] = S[k*2+i, r]
    col += 7
    
    # A1*B3: 2x2*2x3, rank 11
    S223 = _strassen_223()
    v_rows_B3 = [4, 5, 6, 11, 12, 13]  # j*7+k for j in 0-1, k in 4-6
    for r in range(11):
        U[0:4, col+r] = S223[0:4, r]
        for idx, vr in enumerate(v_rows_B3):
            V[vr, col+r] = S223[idx, r]
        for k in range(3):
            for i in range(2):
                W[(k+4)*3+i, col+r] = S223[k*2+i, r]
    col += 11
    
    # A2*B3: 2x2*2x3, rank 11
    v_rows_B3 = [4, 5, 6, 11, 12, 13]
    for r in range(11):
        U[4:8, col+r] = S223[0:4, r]
        for idx, vr in enumerate(v_rows_B3):
            V[vr, col+r] = S223[idx, r]
        for k in range(3):
            for i in range(2):
                W[(k+4)*3+i, col+r] = S223[k*2+i, r]
    col += 11
    
    # 1x4*4x7: rank 4
    # A row 2 (U rows 8-11), B all (V rows 0-27), C row 2 (W rows 6,7,8... wait)
    # W index: k*3+i, k in 0-6, i in 0-2
    # For i=2: W[k*3+2, r]
    for j in range(4):
        U[8+j, col+j] = 1.0
        for k in range(7):
            V[j*7+k, col+j] = 1.0
            W[k*3+2, col+j] = 1.0
    col += 4
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # T_recon[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_expected = _build_tensor(n, m, p)
    if np.max(np.abs(T_recon - T_expected)) > tol:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the block decomposition
    try:
        U, V, W = _decompose_347()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
    
    if time.time() > deadline:
        # Fallback
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
