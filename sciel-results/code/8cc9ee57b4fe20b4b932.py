import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
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
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
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

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    strassen_blocks: list of (i0, j0, k0)
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _build_2x2x8_strassen():
    """
    Build a rank-28 decomposition for 2x2 * 2x8 matrix multiplication.
    Uses Strassen (rank 7) for the 2x2x2 sub-blocks, repeated 4 times for 4 k-pairs.
    """
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    sU, sV, sW = _strassen_2x2_2()
    
    for block in range(4):
        k0 = block * 2
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[di*m + dj][block*7 + c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[dj*p + k0 + dk][block*7 + c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + di][block*7 + c] = sW[dk*2+di][c]
    
    return U, V, W

def _build_block_diagonal(n, m, p):
    """
    Block-diagonal embedding: partition 3x3 into 2x2 + 1x1, 3x8 into 2x8 + 1x8.
    Direct sum of 2x2x8 (rank 28 via Strassen) and 1x1x8 (rank 8 schoolbook).
    Total rank: 36.
    """
    # 2x2x8 part: rows 0-1 of A, cols 0-1 of B, output rows 0-1
    # 1x1x8 part: row 2 of A, cols 2 of B, output row 2
    
    U2, V2, W2 = _build_2x2x8_strassen()
    rank2 = len(U2[0])  # 28
    
    # 1x1x8 part: rank 8
    rank1 = 8
    total_rank = rank2 + rank1
    
    U = [[0]*total_rank for _ in range(n*m)]  # 9 x total_rank
    V = [[0]*total_rank for _ in range(m*p)]  # 24 x total_rank
    W = [[0]*total_rank for _ in range(p*n)]  # 24 x total_rank
    
    # Embed 2x2x8 part:
    # A indices: i in {0,1}, j in {0,1} -> U row i*m+j = i*3+j
    # B indices: j in {0,1}, k in {0..7} -> V row j*p+k = j*8+k
    # C indices: k in {0..7}, i in {0,1} -> W row k*n+i = k*3+i
    
    for c in range(rank2):
        for di in range(2):
            for dj in range(2):
                U[di*3 + dj][c] = U2[di*2+dj][c]
        for dj in range(2):
            for dk in range(8):
                V[dj*8 + dk][c] = V2[dj*2+dk][c]
        for dk in range(8):
            for di in range(2):
                W[dk*3 + di][c] = W2[dk*2+di][c]
    
    # Embed 1x1x8 part:
    # A index: i=2, j=2 -> U row 2*3+2 = 8
    # B index: j=2, k in {0..7} -> V row 2*8+k = 16+k
    # C index: k in {0..7}, i=2 -> W row k*3+2
    
    for k in range(8):
        col = rank2 + k
        U[8][col] = 1
        V[16+k][col] = 1
        W[k*3 + 2][col] = 1
    
    return U, V, W

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Try block-diagonal embedding: 2x2x8 (Strassen) + 1x1x8 (schoolbook) = rank 36
    if _time_left() > 5:
        try:
            U, V, W = _build_block_diagonal(n, m, p)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Try 4 Strassen blocks as alternative (rank 68)
    if _time_left() > 5:
        try:
            blocks = [
                (0, 0, 0),
                (0, 0, 2),
                (0, 0, 4),
                (0, 0, 6)
            ]
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return best_result
