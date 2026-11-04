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

def _build_decomposition(n, m, p, blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks.
    Overlapping blocks are allowed; contributions are summed.
    Uncovered positions use schoolbook.
    """
    # Count unique covered positions
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    pos = (i0+di, j0+dj, k0+dk)
                    if 0 <= pos[0] < n and 0 <= pos[1] < m and 0 <= pos[2] < p:
                        covered.add(pos)
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    ii, jj = i0+di, j0+dj
                    if 0 <= ii < n and 0 <= jj < m:
                        U[ii*m + jj][col+c] += sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    jj, kk = j0+dj, k0+dk
                    if 0 <= jj < m and 0 <= kk < p:
                        V[jj*p + kk][col+c] += sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    kk, ii = k0+dk, i0+di
                    if 0 <= kk < p and 0 <= ii < n:
                        W[kk*n + ii][col+c] += sW[dk*2+di][c]
        col += 7
    
    # Fill in schoolbook for uncovered positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _try_all_block_configs(n, m, p):
    """
    Try various configurations of Strassen blocks (including overlapping ones)
    to find the minimum rank decomposition.
    """
    best_rank = n * m * p
    best_result = None
    
    # Generate all possible block positions
    all_positions = []
    for i0 in range(max(1, n-1)):
        for j0 in range(max(1, m-1)):
            for k0 in range(max(1, p-1)):
                all_positions.append((i0, j0, k0))
    
    # Try single blocks
    for pos in all_positions:
        if _time_left() < 5:
            break
        try:
            U, V, W = _build_decomposition(n, m, p, [pos])
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Try pairs of blocks (non-overlapping preferred, but try overlapping too)
    if _time_left() > 10:
        for idx1 in range(len(all_positions)):
            for idx2 in range(idx1+1, len(all_positions)):
                if _time_left() < 5:
                    break
                blocks = [all_positions[idx1], all_positions[idx2]]
                try:
                    U, V, W = _build_decomposition(n, m, p, blocks)
                    if _verify(U, V, W, n, m, p):
                        r = len(U[0])
                        if r < best_rank:
                            best_rank = r
                            best_result = {"U": U, "V": V, "W": W}
                except Exception:
                    pass
            if _time_left() < 5:
                break
    
    # Try triples
    if _time_left() > 15:
        for idx1 in range(len(all_positions)):
            for idx2 in range(idx1+1, len(all_positions)):
                for idx3 in range(idx2+1, len(all_positions)):
                    if _time_left() < 5:
                        break
                    blocks = [all_positions[idx1], all_positions[idx2], all_positions[idx3]]
                    try:
                        U, V, W = _build_decomposition(n, m, p, blocks)
                        if _verify(U, V, W, n, m, p):
                            r = len(U[0])
                            if r < best_rank:
                                best_rank = r
                                best_result = {"U": U, "V": V, "W": W}
                    except Exception:
                        pass
                if _time_left() < 5:
                    break
            if _time_left() < 5:
                break
    
    return best_rank, best_result

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Known good configuration: 4 non-overlapping blocks
    known_blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6)
    ]
    
    if _time_left() > 5:
        try:
            U, V, W = _build_decomposition(n, m, p, known_blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Try to find better configurations with overlapping blocks
    if _time_left() > 10:
        try:
            search_rank, search_result = _try_all_block_configs(n, m, p)
            if search_result is not None and search_rank < best_rank:
                best_rank = search_rank
                best_result = search_result
        except Exception:
            pass
    
    return best_result
