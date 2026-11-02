"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def _strassen_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W

def _block_2x2x4():
    U1, V1, W1 = _strassen_2x2x2()
    U = [[0]*14 for _ in range(4)]
    V = [[0]*14 for _ in range(8)]
    W = [[0]*14 for _ in range(8)]
    for c in range(7):
        for i in range(4):
            U[i][c] = U1[i][c]
            W[i][c] = W1[i][c]
        for j in range(2):
            V[j*4+0][c] = V1[j*2+0][c]
            V[j*4+1][c] = V1[j*2+1][c]
    for c in range(7):
        for i in range(4):
            U[i][c+7] = U1[i][c]
            W[i+4][c+7] = W1[i][c]
        for j in range(2):
            V[j*4+2][c+7] = V1[j*2+0][c]
            V[j*4+3][c+7] = V1[j*2+1][c]
    return U, V, W

def _block_2x1x4():
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(4)]
    W = [[0]*2 for _ in range(8)]
    U[0][0] = 1
    U[1][1] = 1
    for k in range(4):
        V[k][0] = 1
        V[k][1] = 1
        W[k*2+0][0] = 1
        W[k*2+1][1] = 1
    return U, V, W

def _block_1x2x4():
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(8)]
    W = [[0]*2 for _ in range(4)]
    U[0][0] = 1
    U[1][1] = 1
    for k in range(4):
        V[0*4+k][0] = 1
        V[1*4+k][1] = 1
        W[k][0] = 1
        W[k][1] = 1
    return U, V, W

def _block_1x1x4():
    U = [[1]]
    V = [[1], [1], [1], [1]]
    W = [[1], [1], [1], [1]]
    return U, V, W

def _verify(U, V, W, n, m, p):
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False, f"({i},{j},{k}): got {val}, expected 1.0"
    return True, ""

def construct():
    n, m, p = 3, 4, 8
    deadline = _start_time + 120
    
    row_splits = [(0, 2), (2, 3)]
    col_splits = [(0, 2), (2, 4)]
    depth_splits = [(0, 4), (4, 8)]
    
    U_full = [[0] * 38 for _ in range(n * m)]
    V_full = [[0] * 38 for _ in range(m * p)]
    W_full = [[0] * 38 for _ in range(p * n)]
    
    col_offset = 0
    for (r0, r1) in row_splits:
        for (c0, c1) in col_splits:
            for (d0, d1) in depth_splits:
                a = r1 - r0
                b = c1 - c0
                c = d1 - d0
                if a == 2 and b == 2 and c == 4:
                    Ub, Vb, Wb = _block_2x2x4()
                    rank_b = 14
                elif a == 2 and b == 1 and c == 4:
                    Ub, Vb, Wb = _block_2x1x4()
                    rank_b = 2
                elif a == 1 and b == 2 and c == 4:
                    Ub, Vb, Wb = _block_1x2x4()
                    rank_b = 2
                elif a == 1 and b == 1 and c == 4:
                    Ub, Vb, Wb = _block_1x1x4()
                    rank_b = 1
                else:
                    rank_b = a * b * c
                    Ub = [[0]*rank_b for _ in range(a*b)]
                    Vb = [[0]*rank_b for _ in range(b*c)]
                    Wb = [[0]*rank_b for _ in range(c*a)]
                    col = 0
                    for ii in range(a):
                        for jj in range(b):
                            for kk in range(c):
                                Ub[ii*b+jj][col] = 1
                                Vb[jj*c+kk][col] = 1
                                Wb[kk*a+ii][col] = 1
                                col += 1
                for ii in range(a):
                    for jj in range(b):
                        gi = r0 + ii
                        gj = c0 + jj
                        for rr in range(rank_b):
                            U_full[gi * m + gj][col_offset + rr] = Ub[ii * b + jj][rr]
                for jj in range(b):
                    for kk in range(c):
                        gj = c0 + jj
                        gk = d0 + kk
                        for rr in range(rank_b):
                            V_full[gj * p + gk][col_offset + rr] = Vb[jj * c + kk][rr]
                for kk in range(c):
                    for ii in range(a):
                        gk = d0 + kk
                        gi = r0 + ii
                        for rr in range(rank_b):
                            W_full[gk * n + gi][col_offset + rr] = Wb[kk * a + ii][rr]
                col_offset += rank_b
    
    ok, msg = _verify(U_full, V_full, W_full, n, m, p)
    if not ok:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
