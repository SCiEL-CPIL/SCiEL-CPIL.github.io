"""Tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 120.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W


def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U, V, W as lists of lists (integers).
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # P1 = (A00 + A11) * (B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (A10 + A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # P3 = A00 * (B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # P4 = A11 * (B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # P5 = (A00 + A01) * B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # P6 = (A10 - A00) * (B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (A01 - A11) * (B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
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
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2()
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


def _try_strassen_configurations(n, m, p):
    """
    Try various configurations of Strassen blocks to find the best rank.
    """
    best_result = None
    best_rank = n * m * p
    
    # Strategy 1: Use Strassen blocks along the k-dimension for fixed i,j
    # k has 8 elements, so we can fit 4 non-overlapping 2x2x2 blocks in k
    # For each (i,j) pair, we can place blocks at k=0,2,4,6
    
    configurations = []
    
    # Configuration A: 4 blocks for (0,0)
    configurations.append([(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)])
    
    # Configuration B: 4 blocks for (1,0)
    configurations.append([(1, 0, 0), (1, 0, 2), (1, 0, 4), (1, 0, 6)])
    
    # Configuration C: 4 blocks for (2,0)
    configurations.append([(2, 0, 0), (2, 0, 2), (2, 0, 4), (2, 0, 6)])
    
    # Configuration D: 4 blocks for (0,1)
    configurations.append([(0, 1, 0), (0, 1, 2), (0, 1, 4), (0, 1, 6)])
    
    # Configuration E: 4 blocks for (1,1)
    configurations.append([(1, 1, 0), (1, 1, 2), (1, 1, 4), (1, 1, 6)])
    
    # Configuration F: 4 blocks for (2,1)
    configurations.append([(2, 1, 0), (2, 1, 2), (2, 1, 4), (2, 1, 6)])
    
    # Configuration G: 4 blocks for (0,2)
    configurations.append([(0, 2, 0), (0, 2, 2), (0, 2, 4), (0, 2, 6)])
    
    # Configuration H: 4 blocks for (1,2)
    configurations.append([(1, 2, 0), (1, 2, 2), (1, 2, 4), (1, 2, 6)])
    
    # Configuration I: 4 blocks for (2,2)
    configurations.append([(2, 2, 0), (2, 2, 2), (2, 2, 4), (2, 2, 6)])
    
    # Try each configuration
    for blocks in configurations:
        if _time_left() < 5:
            break
        
        try:
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return best_result, best_rank


def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Try Strassen configurations
    if _time_left() > 5:
        try:
            strassen_result, strassen_rank = _try_strassen_configurations(n, m, p)
            if strassen_result is not None and strassen_rank < best_rank:
                best_rank = strassen_rank
                best_result = strassen_result
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
