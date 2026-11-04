import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _schoolbook_block(n, m, p):
    """Return U, V, W lists for n x m x p schoolbook decomposition."""
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U[i*2+j] corresponds to A_{ij} (i,j in 0..1)
    V[j*2+k] corresponds to B_{jk}
    W[k*2+i] corresponds to C_{ki}
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    # A11 is row 0, col 0 -> index 0
    # A22 is row 1, col 1 -> index 3
    # B11 is row 0, col 0 -> index 0
    # B22 is row 1, col 1 -> index 3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C11 = M1 + M4 - M5 + M7
    # C11 is row 0, col 0 -> index 0
    # C22 is row 1, col 1 -> index 3
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    # A21 is row 1, col 0 -> index 2
    # A22 is row 1, col 1 -> index 3
    # B11 is index 0
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 = M2 + M4
    # C21 is row 1, col 0 -> index 2
    W[2][1] = 1
    
    # M3 = A11(B21-B22)
    # A11 is index 0
    # B21 is row 1, col 0 -> index 2
    # B22 is index 3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    # C12 = M3 + M5
    # C12 is row 0, col 1 -> index 1
    W[1][2] = 1
    
    # M4 = A22(B12-B11)
    # A22 is index 3
    # B12 is row 0, col 1 -> index 1
    # B11 is index 0
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    # C11 += M4, C21 += M4
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A11+A12)B22
    # A11 is index 0
    # A12 is row 0, col 1 -> index 1
    # B22 is index 3
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 -= M5, C12 += M5
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21 is index 2
    # A11 is index 0
    # B11 is index 0
    # B12 is index 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 += M6
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12 is index 1
    # A22 is index 3
    # B21 is index 2
    # B22 is index 3
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 += M7
    W[0][6] = 1
    
    return U, V, W

def _combine_decompositions(blocks):
    """
    blocks: list of (n, m, p, U, V, W)
    Returns combined U, V, W
    """
    total_rank = sum(len(b[2]) for b in blocks)
    U = [[0] * total_rank for _ in range(3 * 5)]
    V = [[0] * total_rank for _ in range(5 * 7)]
    W = [[0] * total_rank for _ in range(7 * 3)]
    
    rank_offset = 0
    for n, m, p, bU, bV, bW in blocks:
        r = len(bW)
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for r_idx in range(r):
                    val = bU[u_idx][r_idx]
                    if val != 0:
                        U[u_idx][rank_offset + r_idx] += val
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                for r_idx in range(r):
                    val = bV[v_idx][r_idx]
                    if val != 0:
                        V[v_idx][rank_offset + r_idx] += val
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                for r_idx in range(r):
                    val = bW[w_idx][r_idx]
                    if val != 0:
                        W[w_idx][rank_offset + r_idx] += val
        rank_offset += r
        
    return U, V, W

def construct():
    start_time = time.time()
    
    # Split n=3 into [2, 1]
    # Split m=5 into [2, 2, 1]
    # Split p=7 into [2, 2, 2, 1]
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2, 1]
    
    blocks = []
    
    for n in n_splits:
        for m in m_splits:
            for p in p_splits:
                if n == 2 and m == 2 and p == 2:
                    U, V, W = _strassen_2x2x2()
                    blocks.append((n, m, p, U, V, W))
                else:
                    U, V, W = _schoolbook_block(n, m, p)
                    blocks.append((n, m, p, U, V, W))
                    
    U, V, W = _combine_decompositions(blocks)
    
    # Verify rank
    rank = len(W[0])
    
    return {"U": U, "V": V, "W": W}
