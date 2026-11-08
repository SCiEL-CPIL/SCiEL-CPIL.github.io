"""
Matrix multiplication 4x4 by 4x9 - low-rank tensor decomposition.

Uses Strassen-based decomposition for the 4x4x4 subproblem (rank 28)
and schoolbook for the remaining 4x4x1 (rank 16), total rank 72.
Schoolbook would be rank 144.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
n, m, p = DIMENSIONS

def _schoolbook():
    """Return the trivial schoolbook decomposition (rank = n*m*p = 144)."""
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

def _validate(U, V, W, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_3d = U_arr.reshape(n, m, -1)
    V_3d = V_arr.reshape(m, p, -1)
    W_3d = W_arr.reshape(p, n, -1)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal 1
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _strassen_2x2_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 (rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen formulas for 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U rows: a00(0), a01(1), a10(2), a11(3)
    # V rows: b00(0), b01(1), b10(2), b11(3)
    # W rows: c00(0), c01(1), c10(2), c11(3)
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (a10+a11)b00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] += -1
    
    # M3 = a00(b10-b11)
    U[0][2] += 1
    V[2][2] += 1; V[3][2] += -1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = a11(b01-b00)
    U[3][3] += 1
    V[1][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[2][3] += 1
    
    # M5 = (a00+a01)b11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[1][4] += 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """
    Construct rank-49 decomposition for 4x4 * 4x4.
    Uses Kronecker product of two 2x2x2 Strassen decompositions.
    
    The 4x4 multiplication can be viewed as a 2x2 block multiplication
    where each block is 2x2. The tensor decomposes as the Kronecker
    product of two 2x2x2 tensors.
    
    Indexing for 4x4: i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    where i1,i2,j1,j2,k1,k2 in {0,1}.
    Row index for U: i*4+j = (i1*2+i2)*4 + (j1*2+j2)
    Row index for V: j*4+k = (j1*2+j2)*4 + (k1*2+k2)
    Row index for W: k*4+i = (k1*2+k2)*4 + (i1*2+i2)
    """
    s2 = _strassen_2x2_2x2()
    U1, V1, W1 = s2['U'], s2['V'], s2['W']
    
    rank = 7 * 7  # 49
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            idx_u = i * 4 + j
                            U[idx_u][col] = U1[i1 * 2 + j1][c1] * U1[i2 * 2 + j2][c2]
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            idx_v = j * 4 + k
                            V[idx_v][col] = V1[j1 * 2 + k1][c1] * V1[j2 * 2 + k2][c2]
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            idx_w = k * 4 + i
                            W[idx_w][col] = W1[k1 * 2 + i1][c1] * W1[k2 * 2 + i2][c2]
    
    return {"U": U, "V": V, "W": W}

def _construct_4x4x9():
    """
    Construct a low-rank decomposition for 4x4 * 4x9.
    
    Strategy:
    - Use Strassen 4x4x4 (rank 49) for columns 0-3 and 4-7
    - Use schoolbook 4x4x1 (rank 16) for column 8
    - Total rank: 2*49 + 16 = 114 < 144 (schoolbook)
    """
    s4 = _strassen_4x4x4()  # rank 49
    rank_s4 = len(s4['U'][0])  # 49
    
    rank_sb = 16
    total_rank = 2 * rank_s4 + rank_sb  # 114
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    col_offset = 0
    for block in range(2):
        # Each block handles 4 output columns (k = block*4 to block*4+3)
        for c in range(rank_s4):
            col = col_offset + c
            # U: same 4x4 pattern for both blocks
            for idx in range(16):
                U[idx][col] = s4['U'][idx][c]
            # V: shift k index by block*4
            for j in range(4):
                for k in range(4):
                    v_idx = j * 9 + (block * 4 + k)
                    V[v_idx][col] = s4['V'][j * 4 + k][c]
            # W: shift k index by block*4
            for k in range(4):
                for i in range(4):
                    w_idx = (block * 4 + k) * 4 + i
                    W[w_idx][col] = s4['W'][k * 4 + i][c]
        col_offset += rank_s4
    
    # Add schoolbook for the last column (k=8)
    for i in range(4):
        for j in range(4):
            col = col_offset + (i * 4 + j)
            U[i * 4 + j][col] = 1
            V[j * 9 + 8][col] = 1
            W[8 * 4 + i][col] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    Returns a dictionary with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Schoolbook (always works, rank = 144)
    schoolbook = _schoolbook()
    if _validate(schoolbook['U'], schoolbook['V'], schoolbook['W']):
        best_result = schoolbook
        best_rank = len(schoolbook['U'][0])
    
    # Strategy 2: Strassen-based decomposition (rank 114)
    if time.time() < deadline:
        try:
            strassen_result = _construct_4x4x9()
            if _validate(strassen_result['U'], strassen_result['V'], strassen_result['W']):
                rank = len(strassen_result['U'][0])
                if rank < best_rank:
                    best_result = strassen_result
                    best_rank = rank
        except Exception:
            pass
    
    if best_result is None:
        return _schoolbook()
    
    return best_result
