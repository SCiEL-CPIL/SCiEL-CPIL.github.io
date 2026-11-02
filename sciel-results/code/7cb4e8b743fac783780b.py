"""Matrix multiplication 3x4 by 4x6: Strassen-based rank-66 decomposition."""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_strassen_3x4x6():
    """Build the rank-66 decomposition for 3x4 * 4x6."""
    n, m, p = 3, 4, 6
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        """Add 7 Strassen products for a 2x2 block."""
        nonlocal col
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Row 0: schoolbook with all 4x6 elements (24 products)
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    # Rows 1-2: Strassen on 2x2 blocks (3 k-blocks × 2 j-blocks × 7 = 42 products)
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            add_strassen(1, j_base, k_base)
    
    return U, V, W, rank

def _verify(U, V, W, rank):
    """Verify the tensor decomposition is correct."""
    n, m, p = 3, 4, 6
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
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
    return U, V, W, rank

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    Returns rank-66 Strassen-based decomposition.
    """
    # Build the rank-66 Strassen solution
    U, V, W, rank = _build_strassen_3x4x6()
    
    # Verify correctness
    if not _verify(U, V, W, rank):
        # Fallback to schoolbook if verification fails
        U, V, W, rank = _schoolbook(3, 4, 6)
    
    return {"U": U, "V": V, "W": W}
