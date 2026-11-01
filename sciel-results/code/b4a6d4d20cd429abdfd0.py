"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _build_decomp(n, m, p):
    """
    Build a decomposition using block structure.
    Rank 44: Strassen for 2x2x8 part (28) + Schoolbook for 2x1x8 part (16).
    """
    n, m, p = 2, 3, 8
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A2*B2 (rank 16)
    for k in range(8):
        # C[0,k] += A[0,2] * B[2,k]
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        # C[1,k] += A[1,2] * B[2,k]
        U[5][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Part 2: A1*B1 via Strassen (rank 28)
    for g in range(4):
        b11 = 0 + 2 * g      
        b12 = 0 + 2 * g + 1  
        b21 = 8 + 2 * g      
        b22 = 8 + 2 * g + 1  
        
        a11 = 0  
        a12 = 1  
        a21 = 3  
        a22 = 4  
        
        # M1
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        W[(2 * g) * 2 + 0][col] = 1      
        W[(2 * g + 1) * 2 + 1][col] = 1  
        col += 1
        
        # M2
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        W[(2 * g) * 2 + 1][col] = 1  
        col += 1
        
        # M3
        U[a11][col] = 1
        V[b22][col] = 1
        V[b12][col] = -1
        W[(2 * g + 1) * 2 + 0][col] = 1   
        W[(2 * g + 1) * 2 + 1][col] = -1  
        col += 1
        
        # M4
        U[a22][col] = 1
        V[b11][col] = 1
        V[b21][col] = -1
        W[(2 * g) * 2 + 0][col] = -1  
        W[(2 * g) * 2 + 1][col] = 1   
        col += 1
        
        # M5
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        W[(2 * g + 1) * 2 + 0][col] = 1  
        col += 1
        
        # M6
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        W[(2 * g) * 2 + 0][col] = 1  
        col += 1
        
        # M7
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        W[(2 * g + 1) * 2 + 1][col] = 1  
        col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    Condition: sum_c U[i, j1, c] * V[j2, k, c] * W[k, i, c] == delta(j1, j2)
    """
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    max_err = 0.0
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    expected = 1.0 if j1 == j2 else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j1, c] * V[j2 * p + k, c] * W[k * n + i, c]
                    err = abs(val - expected)
                    if err > max_err:
                        max_err = err
    return max_err


def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition (rank 44)
    try:
        U, V, W = _build_decomp(n, m, p)
        if time.time() < deadline:
            err = _verify(U, V, W, n, m, p)
            if err < 1e-10:
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    if time.time() < deadline:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
        
    # If we somehow run out of time (unlikely for this small problem), return empty or raise
    # But construct must return a dict.
    return {"U": [], "V": [], "W": []}

# EVOLVE-BLOCK-END
