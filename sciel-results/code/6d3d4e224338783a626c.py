import time
import numpy as np

DIMENSIONS = [5, 5, 5]
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _strassen_2x2_block(U, V, W, n, m, p, i_start, j_start, k_start, col):
    """Apply Strassen to a 2x2x2 block, return new col."""
    i0, i1 = i_start, i_start + 1
    j0, j1 = j_start, j_start + 1
    k0, k1 = k_start, k_start + 1
    
    # M1
    U[i0 * m + j0][col] = 1
    U[i1 * m + j1][col] = 1
    V[j0 * p + k0][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i0][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1
    
    # M2
    U[i1 * m + j0][col] = 1
    U[i1 * m + j1][col] = 1
    V[j0 * p + k0][col] = 1
    W[k0 * n + i1][col] = 1
    W[k1 * n + i1][col] = -1
    col += 1
    
    # M3
    U[i0 * m + j0][col] = 1
    V[j0 * p + k1][col] = 1
    V[j1 * p + k1][col] = -1
    W[k1 * n + i0][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1
    
    # M4
    U[i1 * m + j1][col] = 1
    V[j1 * p + k0][col] = 1
    V[j0 * p + k0][col] = -1
    W[k0 * n + i0][col] = 1
    W[k0 * n + i1][col] = 1
    col += 1
    
    # M5
    U[i0 * m + j0][col] = 1
    U[i0 * m + j1][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i0][col] = -1
    W[k1 * n + i0][col] = 1
    col += 1
    
    # M6
    U[i1 * m + j0][col] = 1
    U[i0 * m + j0][col] = -1
    V[j0 * p + k0][col] = 1
    V[j0 * p + k1][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1
    
    # M7
    U[i0 * m + j1][col] = 1
    U[i1 * m + j1][col] = -1
    V[j1 * p + k0][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i0][col] = 1
    col += 1
    
    return col

def _schoolbook_block(U, V, W, n, m, p, i_start, j_start, k_start, ni, mj, pk, col):
    """Apply schoolbook to a ni x mj x pk block, return new col."""
    for u in range(ni):
        for v in range(mj):
            for w in range(pk):
                U[(i_start + u) * m + (j_start + v)][col] = 1
                V[(j_start + v) * p + (k_start + w)][col] = 1
                W[(k_start + w) * n + (i_start + u)][col] = 1
                col += 1
    return col

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Partitions
    n_parts = [2, 2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    n_starts = [0, 2, 4]
    m_starts = [0, 2, 4]
    p_starts = [0, 2, 4]
    
    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if n_parts[i] == 2 and m_parts[j] == 2 and p_parts[k] == 2:
                    total_rank += 7
                else:
                    total_rank += n_parts[i] * m_parts[j] * p_parts[k]
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() - t0 > _DEADLINE:
                    # Return whatever we have so far (may be invalid, but better than timeout)
                    break
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    col = _strassen_2x2_block(U, V, W, n, m, p, i_start, j_start, k_start, col)
                else:
                    col = _schoolbook_block(U, V, W, n, m, p, i_start, j_start, k_start, ni, mj, pk, col)
    
    return {"U": U, "V": V, "W": W}
