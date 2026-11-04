import time
import numpy as np

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2(U, V, W, rank_offset, 
                  a_rows, a_cols, 
                  b_rows, b_cols, 
                  c_rows, c_cols):
    """
    Adds Strassen decomposition for C = A B.
    A is a_rows x a_cols (2x2)
    B is b_rows x b_cols (2x2)
    C is c_rows x c_cols (2x2)
    Indices are global.
    """
    # A indices: (a_rows[0]+r, a_cols[0]+c)
    # B indices: (b_rows[0]+r, b_cols[0]+c)
    # C indices: (c_rows[0]+r, c_cols[0]+c)
    
    # Strassen Multiplications
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def get_U_idx(r, c):
        return (a_rows[r] * 5 + a_cols[c]) # n=2, m=5. U index is i*m+j. i=a_rows[r], j=a_cols[c]
        
    def get_V_idx(r, c):
        return (b_rows[r] * 5 + b_cols[c]) # m=5, p=5. V index is j*p+k. j=b_rows[r], k=b_cols[c]
        
    def get_W_idx(r, c):
        return (c_cols[c] * 2 + c_rows[r]) # p=5, n=2. W index is k*n+i. k=c_cols[c], i=c_rows[r]

    # We need to add to U, V, W at rank_offset .. rank_offset+6
    
    # M1
    idx = rank_offset
    # U part: A00 + A11
    U[get_U_idx(0,0)][idx] += 1
    U[get_U_idx(1,1)][idx] += 1
    # V part: B00 + B11
    V[get_V_idx(0,0)][idx] += 1
    V[get_V_idx(1,1)][idx] += 1
    # W part: C00 + C11 (wait, W is output. M1 contributes to C00 and C11)
    # C00 gets +1, C11 gets +1
    W[get_W_idx(0,0)][idx] += 1
    W[get_W_idx(1,1)][idx] += 1
    
    # M2
    idx = rank_offset + 1
    # U: A10 + A11
    U[get_U_idx(1,0)][idx] += 1
    U[get_U_idx(1,1)][idx] += 1
    # V: B00
    V[get_V_idx(0,0)][idx] += 1
    # W: C10 gets +1
    W[get_W_idx(1,0)][idx] += 1
    
    # M3
    idx = rank_offset + 2
    # U: A00
    U[get_U_idx(0,0)][idx] += 1
    # V: B01 - B11
    V[get_V_idx(0,1)][idx] += 1
    V[get_V_idx(1,1)][idx] -= 1
    # W: C01 gets +1
    W[get_W_idx(0,1)][idx] += 1
    
    # M4
    idx = rank_offset + 3
    # U: A11
    U[get_U_idx(1,1)][idx] += 1
    # V: B10 - B00
    V[get_V_idx(1,0)][idx] += 1
    V[get_V_idx(0,0)][idx] -= 1
    # W: C00 gets +1, C10 gets +1
    W[get_W_idx(0,0)][idx] += 1
    W[get_W_idx(1,0)][idx] += 1
    
    # M5
    idx = rank_offset + 4
    # U: A00 + A01
    U[get_U_idx(0,0)][idx] += 1
    U[get_U_idx(0,1)][idx] += 1
    # V: B11
    V[get_V_idx(1,1)][idx] += 1
    # W: C00 gets -1, C01 gets +1
    W[get_W_idx(0,0)][idx] -= 1
    W[get_W_idx(0,1)][idx] += 1
    
    # M6
    idx = rank_offset + 5
    # U: A10 - A00
    U[get_U_idx(1,0)][idx] += 1
    U[get_U_idx(0,0)][idx] -= 1
    # V: B00 + B01
    V[get_V_idx(0,0)][idx] += 1
    V[get_V_idx(0,1)][idx] += 1
    # W: C11 gets +1
    W[get_W_idx(1,1)][idx] += 1
    
    # M7
    idx = rank_offset + 6
    # U: A01 - A11
    U[get_U_idx(0,1)][idx] += 1
    U[get_U_idx(1,1)][idx] -= 1
    # V: B10 + B11
    V[get_V_idx(1,0)][idx] += 1
    V[get_V_idx(1,1)][idx] += 1
    # W: C00 gets +1
    W[get_W_idx(0,0)][idx] += 1
    
    return rank_offset + 7

def _schoolbook_2x2_2x1(U, V, W, rank_offset,
                        a_rows, a_cols,
                        b_rows, b_cols,
                        c_rows, c_cols):
    """
    C = A B.
    A is 2x2, B is 2x1, C is 2x1.
    Rank 4.
    """
    for r in range(2):
        for c in range(2): # c is col of B (which is row of B effectively, size 1? No, B is 2x1, so 2 rows, 1 col)
             # Wait, B is 2x1. Indices (row, col).
             # A is 2x2.
             # C is 2x1.
             # C_i = sum_j A_ij B_j0.
             # j=0,1. i=0,1.
             pass
             
    # Correct loop:
    # i in 0..1 (row of C, row of A)
    # j in 0..1 (col of A, row of B)
    # k in 0..0 (col of B, col of C)
    
    for i in range(2):
        for j in range(2):
            # k is 0
            idx = rank_offset + i * 2 + j
            U[a_rows[i] * 5 + a_cols[j]][idx] += 1
            V[b_rows[j] * 5 + b_cols[0]][idx] += 1
            W[c_cols[0] * 2 + c_rows[i]][idx] += 1
            
    return rank_offset + 4

def _schoolbook_2x1_1x5(U, V, W, rank_offset,
                        a_rows, a_cols,
                        b_rows, b_cols,
                        c_rows, c_cols):
    """
    C = A B.
    A is 2x1, B is 1x5, C is 2x5.
    Rank 10.
    """
    # i in 0..1 (row of A, row of C)
    # j in 0..0 (col of A, row of B)
    # k in 0..4 (col of B, col of C)
    
    for i in range(2):
        for k in range(5):
            # j is 0
            idx = rank_offset + i * 5 + k
            U[a_rows[i] * 5 + a_cols[0]][idx] += 1
            V[b_rows[0] * 5 + b_cols[k]][idx] += 1
            W[c_cols[k] * 2 + c_rows[i]][idx] += 1
            
    return rank_offset + 10

def _construct_rank_46():
    n, m, p = 2, 5, 5
    total_rank = 46
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # Block 1: j in {0,1}
    # A cols 0,1. B rows 0,1.
    
    # 1.1: k in {0,1}. Strassen.
    rank_offset = _strassen_2x2(U, V, W, rank_offset, 
                                 [0, 1], [0, 1], # A rows, A cols
                                 [0, 1], [0, 1], # B rows, B cols
                                 [0, 1], [0, 1]) # C rows, C cols
    
    # 1.2: k in {2,3}. Strassen.
    rank_offset = _strassen_2x2(U, V, W, rank_offset,
                                 [0, 1], [0, 1],
                                 [0, 1], [2, 3],
                                 [0, 1], [2, 3])
    
    # 1.3: k in {4}. Schoolbook 2x2 * 2x1.
    rank_offset = _schoolbook_2x2_2x1(U, V, W, rank_offset,
                                       [0, 1], [0, 1],
                                       [0, 1], [4],
                                       [0, 1], [4])
    
    # Block 2: j in {2,3}
    # A cols 2,3. B rows 2,3.
    
    # 2.1: k in {0,1}. Strassen.
    rank_offset = _strassen_2x2(U, V, W, rank_offset,
                                 [0, 1], [2, 3],
                                 [2, 3], [0, 1],
                                 [0, 1], [0, 1])
    
    # 2.2: k in {2,3}. Strassen.
    rank_offset = _strassen_2x2(U, V, W, rank_offset,
                                 [0, 1], [2, 3],
                                 [2, 3], [2, 3],
                                 [0, 1], [2, 3])
    
    # 2.3: k in {4}. Schoolbook.
    rank_offset = _schoolbook_2x2_2x1(U, V, W, rank_offset,
                                       [0, 1], [2, 3],
                                       [2, 3], [4],
                                       [0, 1], [4])
    
    # Block 3: j in {4}
    # A col 4. B row 4.
    
    # 3.1: k in {0..4}. Schoolbook 2x1 * 1x5.
    rank_offset = _schoolbook_2x1_1x5(U, V, W, rank_offset,
                                       [0, 1], [4],
                                       [4], [0, 1, 2, 3, 4],
                                       [0, 1], [0, 1, 2, 3, 4])
    
    if rank_offset != total_rank:
        raise ValueError(f"Rank mismatch: {rank_offset} vs {total_rank}")
        
    return {"U": U, "V": V, "W": W}

def _construct():
    n, m, p = DIMENSIONS
    # Try rank 46
    try:
        res = _construct_rank_46()
        if _verify(n, m, p, res["U"], res["V"], res["W"]):
            return res
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
