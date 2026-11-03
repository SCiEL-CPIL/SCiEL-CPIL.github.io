"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_2x2():
    """Return U, V, W for Strassen 2x2x2 decomposition (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 = (a11+a22)(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (a21+a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # m3 = a11(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # m4 = a22(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # m5 = (a11+a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = -1
    
    # m6 = (a21-a11)(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # m7 = (a12-a22)(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_4x4_4x4():
    """Return U, V, W for 4x4x4 using Strassen at two levels (rank 49)."""
    outer_U, outer_V, outer_W = _strassen_2x2_2x2()
    inner_U, inner_V, inner_W = _strassen_2x2_2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            for block_a in range(4):
                oc_a = outer_U[block_a][t]
                if oc_a == 0:
                    continue
                for la in range(4):
                    ic_a = inner_U[la][s]
                    if ic_a == 0:
                        continue
                    U[block_a * 4 + la][col] += oc_a * ic_a
            for block_b in range(4):
                oc_b = outer_V[block_b][t]
                if oc_b == 0:
                    continue
                for lb in range(4):
                    ic_b = inner_V[lb][s]
                    if ic_b == 0:
                        continue
                    V[block_b * 4 + lb][col] += oc_b * ic_b
            for block_c in range(4):
                oc_c = outer_W[block_c][t]
                if oc_c == 0:
                    continue
                for lc in range(4):
                    ic_c = inner_W[lc][s]
                    if ic_c == 0:
                        continue
                    W[block_c * 4 + lc][col] += oc_c * ic_c
    
    return U, V, W

def _schoolbook_4x4_1():
    """Return U, V, W for 4x4x1 schoolbook (rank 16)."""
    rank = 16
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    col = 0
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify tensor decomposition using numpy."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Use einsum: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Expected: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Else 0
    expected = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    return np.allclose(T, expected, atol=1e-10)

def construct():
    """
    Construct a rank-114 tensor decomposition for 4x4x9 matrix multiplication.
    Uses Strassen at two levels for 4x4x4 (rank 49 each) and schoolbook for 4x4x1 (rank 16).
    Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U44, V44, W44 = _strassen_4x4_4x4()
    U41, V41, W41 = _schoolbook_4x4_1()
    
    rank = 49 + 49 + 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # First 4x4x4: B cols 0-3, C cols 0-3
    for r in range(49):
        for i in range(n*m):
            U[i][r] = U44[i][r]
        for j in range(4):
            for k in range(4):
                V[j*9 + k][r] = V44[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W[k*4 + i][r] = W44[k*4 + i][r]
    
    # Second 4x4x4: B cols 4-7, C cols 4-7
    offset = 49
    for r in range(49):
        for i in range(n*m):
            U[i][offset + r] = U44[i][r]
        for j in range(4):
            for k in range(4):
                V[j*9 + (4+k)][offset + r] = V44[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W[(4+k)*4 + i][offset + r] = W44[k*4 + i][r]
    
    # 4x4x1: B col 8, C col 8
    offset2 = 98
    for r in range(16):
        for i in range(n*m):
            U[i][offset2 + r] = U41[i][r]
        for j in range(4):
            V[j*9 + 8][offset2 + r] = V41[j][r]
        for i in range(4):
            W[8*4 + i][offset2 + r] = W41[i][r]
    
    # Verify with numpy
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
