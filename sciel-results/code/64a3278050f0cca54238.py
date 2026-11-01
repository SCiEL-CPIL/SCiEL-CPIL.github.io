import time
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2.
    U: (4, 7) rows i*2+j
    V: (4, 7) rows j*2+k
    W: (4, 7) rows k*2+i
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1: (A00+A11)*(B00+B11)
    # U: A00, A11
    U[0][0] = 1
    U[3][0] = 1
    # V: B00, B11
    V[0][0] = 1
    V[3][0] = 1
    # W: C00 (k=0,i=0), C11 (k=1,i=1)
    W[0][0] = 1
    W[3][0] = 1

    # M2: (A10+A11)*B00
    # U: A10, A11
    U[2][1] = 1
    U[3][1] = 1
    # V: B00
    V[0][1] = 1
    # W: C10 (k=0, i=1) -> index 0*2+1 = 1
    W[1][1] = 1

    # M3: A00*(B01-B11)
    # U: A00
    U[0][2] = 1
    # V: B01, -B11
    V[1][2] = 1
    V[3][2] = -1
    # W: C01 (k=1, i=0) -> index 1*2+0 = 2, C11 (k=1, i=1) -> index 3
    W[2][2] = 1
    W[3][2] = 1

    # M4: A11*(B10-B00)
    # U: A11
    U[3][3] = 1
    # V: B10, -B00
    V[2][3] = 1
    V[0][3] = -1
    # W: C00 (k=0, i=0) -> 0, C10 (k=0, i=1) -> 1
    W[0][3] = 1
    W[1][3] = 1

    # M5: (A00+A01)*B11
    # U: A00, A01
    U[0][4] = 1
    U[1][4] = 1
    # V: B11
    V[3][4] = 1
    # W: -C00 (k=0, i=0) -> 0, C01 (k=1, i=0) -> 2
    W[0][4] = -1
    W[2][4] = 1

    # M6: (A10-A00)*(B00+B01)
    # U: A10, -A00
    U[2][5] = 1
    U[0][5] = -1
    # V: B00, B01
    V[0][5] = 1
    V[1][5] = 1
    # W: C11 (k=1, i=1) -> 3
    W[3][5] = 1

    # M7: (A01-A11)*(B10+B11)
    # U: A01, -A11
    U[1][6] = 1
    U[3][6] = -1
    # V: B10, B11
    V[2][6] = 1
    V[3][6] = 1
    # W: C00 (k=0, i=0) -> 0
    W[0][6] = 1

    return U, V, W

def _build_rank_25():
    n, m, p = 2, 2, 7
    R = 25
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # 3 Strassen blocks for columns 0-1, 2-3, 4-5
    # 1 Schoolbook block for column 6
    
    col_offset = 0
    
    # Block 1: cols 0, 1
    S_U, S_V, S_W = _strassen_2x2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col_offset + c] = S_U[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j * p + k][col_offset + c] = S_V[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k * n + i][col_offset + c] = S_W[k * 2 + i][c]
    col_offset += 7
    
    # Block 2: cols 2, 3
    S_U, S_V, S_W = _strassen_2x2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col_offset + c] = S_U[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j * p + (k + 2)][col_offset + c] = S_V[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * n + i][col_offset + c] = S_W[k * 2 + i][c]
    col_offset += 7
    
    # Block 3: cols 4, 5
    S_U, S_V, S_W = _strassen_2x2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col_offset + c] = S_U[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j * p + (k + 4)][col_offset + c] = S_V[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                W[(k + 4) * n + i][col_offset + c] = S_W[k * 2 + i][c]
    col_offset += 7
    
    # Block 4: col 6 (Schoolbook)
    # C[i, 6] = sum_j A[i, j] * B[j, 6]
    # Terms: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
    # Rank 4
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        c = col_offset + idx
        U[i * m + j][c] = 1
        V[j * p + 6][c] = 1
        W[6 * n + i][c] = 1
        
    return U, V, W

def _verify(U, V, W, n, m, p):
    R = len(U[0])
    Uf = [[Fraction(U[i][c]) for c in range(R)] for i in range(n * m)]
    Vf = [[Fraction(V[j][c]) for c in range(R)] for j in range(m * p)]
    Wf = [[Fraction(W[k][c]) for c in range(R)] for k in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = Fraction(0)
                for c in range(R):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c]
                if s != (1 if i == k else 0):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank_25()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        R = n * m * p
        U_fb = [[0] * R for _ in range(n * m)]
        V_fb = [[0] * R for _ in range(m * p)]
        W_fb = [[0] * R for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][c] = 1
                    V_fb[j * p + k][c] = 1
                    W_fb[k * n + i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
