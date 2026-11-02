"""Construct a rank-34 tensor decomposition for 3x3 * 3x4 matrix multiplication using Strassen blocks."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(n, m, p, U, V, W):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)"""
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(Ua[i*m+j, rr] * Va[j*p+k, rr] * Wa[k*n+i, rr] for rr in range(r))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank34_strassen():
    """Build rank-34 decomposition: 2 Strassen blocks (2x2x2) + schoolbook for remaining parts.
    
    Strategy:
    - Split the 3x3x4 multiplication into:
      * Top-left 2x2x2 block (i=0,1; j=0,1; k=0,1): Strassen rank 7
      * Top-right 2x2x2 block (i=0,1; j=0,1; k=2,3): Strassen rank 7
      * Bottom row (i=2; j=0,1,2; k=0,1,2,3): schoolbook rank 12
      * Right column (i=0,1; j=2; k=0,1,2,3): schoolbook rank 8
    Total: 7 + 7 + 12 + 8 = 34
    """
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a standard multiplication term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Helper to add a Strassen block for 2x2x2
    # local_i, local_j, local_k correspond to offsets in global i, j, k
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # 7 Strassen terms for 2x2x2 matrix multiplication
        
        # Term 1: M1 = (A11+A22) * (B11+B22)
        # A: (i_off, j_off) + (i_off+1, j_off+1)
        # B: (j_off, k_off) + (j_off+1, k_off+1)
        # C: C11 += M1, C22 += M1
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1       # C11
        W[(k_off+1) * n + (i_off+1)][col] = 1 # C22
        col += 1

        # Term 2: M2 = (A21+A22) * B11
        # A: (i_off+1, j_off) + (i_off+1, j_off+1)
        # B: (j_off, k_off)
        # C: C21 += M2, C22 -= M2
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1   # C21
        W[(k_off+1) * n + (i_off+1)][col] = -1 # C22
        col += 1

        # Term 3: M3 = A11 * (B12-B22)
        # A: (i_off, j_off)
        # B: (j_off, k_off+1) - (j_off+1, k_off+1)
        # C: C12 += M3, C22 += M3
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = 1   # C12
        W[(k_off+1) * n + (i_off+1)][col] = 1 # C22
        col += 1

        # Term 4: M4 = A22 * (B21-B11)
        # A: (i_off+1, j_off+1)
        # B: (j_off+1, k_off) - (j_off, k_off)
        # C: C11 += M4, C21 += M4
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1       # C11
        W[k_off * n + (i_off+1)][col] = 1   # C21
        col += 1

        # Term 5: M5 = (A11+A12) * B22
        # A: (i_off, j_off) + (i_off, j_off+1)
        # B: (j_off+1, k_off+1)
        # C: C11 -= M5, C12 += M5
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1     # C11
        W[(k_off+1) * n + i_off][col] = 1  # C12
        col += 1

        # Term 6: M6 = (A21-A11) * (B11+B12)
        # A: (i_off+1, j_off) - (i_off, j_off)
        # B: (j_off, k_off) + (j_off, k_off+1)
        # C: C22 += M6
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1 # C22
        col += 1

        # Term 7: M7 = (A12-A22) * (B21+B22)
        # A: (i_off, j_off+1) - (i_off+1, j_off+1)
        # B: (j_off+1, k_off) + (j_off+1, k_off+1)
        # C: C11 += M7
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1       # C11
        col += 1

    # 1. Top-left 2x2x2 block (i=0,1; j=0,1; k=0,1): Strassen rank 7
    add_strassen(0, 0, 0)
    
    # 2. Top-right 2x2x2 block (i=0,1; j=0,1; k=2,3): Strassen rank 7
    add_strassen(0, 0, 2)
    
    # 3. Bottom row (i=2; j=0,1,2; k=0,1,2,3): schoolbook rank 12
    for j in range(3):
        for k in range(4):
            add_term(2 * m + j, j * p + k, k * n + 2)
    
    # 4. Right column (i=0,1; j=2; k=0,1,2,3): schoolbook rank 8
    for i in range(2):
        for k in range(4):
            add_term(i * m + 2, 2 * p + k, k * n + i)
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    """Build schoolbook rank-36 decomposition."""
    n, m, p = 3, 3, 4
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
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = 3, 3, 4
    
    # Try rank-34 Strassen-based decomposition
    try:
        result = _build_rank34_strassen()
        if _verify(n, m, p, result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    if time.time() > deadline:
        return _build_schoolbook()
    
    # Fallback: schoolbook rank 36
    return _build_schoolbook()
# EVOLVE-BLOCK-END
