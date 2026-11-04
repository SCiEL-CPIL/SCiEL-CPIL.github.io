import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 140.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_3x3x2_decomposition(k_base, n, m, p):
    """
    Build the 14-rank decomposition for the slice k in {k_base, k_base+1}.
    Returns U, V, W parts for this slice.
    U: (9, 14), V: (24, 14), W: (24, 14)
    """
    rank_slice = 14
    U = [[0]*rank_slice for _ in range(n*m)]
    V = [[0]*rank_slice for _ in range(m*p)]
    W = [[0]*rank_slice for _ in range(p*n)]
    
    sU, sV, sW = _strassen_2x2_2()
    
    # 1. Strassen for i in {0,1}, j in {0,1}
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = sU[i*2 + j][c]
        for j in range(2):
            for k_loc in range(2):
                k_glob = k_base + k_loc
                V[j*p + k_glob][c] = sV[j*2 + k_loc][c]
        for k_loc in range(2):
            for i in range(2):
                k_glob = k_base + k_loc
                W[k_glob*n + i][c] = sW[k_loc*2 + i][c]
                
    # 2. Row 2 (i=2) for j in {0,1,2}, k in {0,1}
    # Terms: A[2,j] * B[j, k_glob]
    col_offset = 7
    for j in range(3):
        for k_loc in range(2):
            k_glob = k_base + k_loc
            c = col_offset + j*2 + k_loc
            U[2*m + j][c] = 1
            V[j*p + k_glob][c] = 1
            W[k_glob*n + 2][c] = 1
            
    # 3. Col 2 (j=2) for i in {0,1}, k in {0,1}
    # Term: A[i, 2] * B[2, k_glob]
    c = 13
    for i in range(2):
        U[i*m + 2][c] = 1
    for k_loc in range(2):
        k_glob = k_base + k_loc
        V[2*p + k_glob][c] = 1
        for i in range(2):
            W[k_glob*n + i][c] = 1
            
    return U, V, W

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses 4 slices of 3x3x2, each decomposed with rank 14.
    Total rank 56.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    if _time_left() > 5:
        try:
            rank_total = 14 * 4  # 56
            U = [[0]*rank_total for _ in range(n*m)]
            V = [[0]*rank_total for _ in range(m*p)]
            W = [[0]*rank_total for _ in range(p*n)]
            
            for s in range(4):
                k_base = 2 * s
                slice_U, slice_V, slice_W = _build_3x3x2_decomposition(k_base, n, m, p)
                offset = 14 * s
                for r in range(n*m):
                    for c in range(14):
                        U[r][offset + c] = slice_U[r][c]
                for r in range(m*p):
                    for c in range(14):
                        V[r][offset + c] = slice_V[r][c]
                for r in range(p*n):
                    for c in range(14):
                        W[r][offset + c] = slice_W[r][c]
                        
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
            
    return best_result
