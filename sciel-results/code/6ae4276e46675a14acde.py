"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return (U, V, W) for Strassen 2x2x2 multiplication, rank 7."""
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # U row (i,j):
    U[0][0] = 1.0; U[0][2] = 1.0; U[0][4] = 1.0; U[0][5] = -1.0; U[0][6] = -1.0
    U[1][2] = 1.0; U[1][4] = 1.0; U[1][5] = -1.0; U[1][6] = -1.0
    U[2][0] = 1.0; U[2][1] = 1.0; U[2][5] = 1.0
    U[3][0] = 1.0; U[3][1] = 1.0; U[3][3] = 1.0; U[3][6] = 1.0
    
    # V row (j,k):
    V[0][0] = 1.0; V[0][1] = 1.0; V[0][5] = 1.0
    V[1][0] = 1.0; V[1][3] = -1.0; V[1][5] = 1.0
    V[2][2] = 1.0; V[2][4] = 1.0; V[2][6] = 1.0
    V[3][0] = 1.0; V[3][3] = 1.0; V[3][4] = 1.0; V[3][6] = 1.0
    
    # W row (k,i):
    W[0][0] = 1.0; W[0][3] = 1.0; W[0][4] = -1.0; W[0][6] = -1.0
    W[1][2] = 1.0; W[1][4] = 1.0
    W[2][1] = 1.0; W[2][3] = 1.0
    W[3][0] = 1.0; W[3][1] = -1.0; W[3][2] = 1.0; W[3][6] = 1.0
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta(i,j,k) for all i,j,k."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if nm > 0 else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i, j, k) == (0, 0, 0) else 0.0
                u_vals = U_arr[i * m + j, :]
                v_vals = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                total = float(np.dot(np.multiply(u_vals, v_vals), w_val))
                if abs(total - expected) > 1e-6:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 120.0
    MARGIN = 10.0
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Define block splits
    # i: 2, 1
    # j: 2, 2, 1
    # k: 2, 2, 1
    
    i_blocks = [[0, 1], [2]]
    j_blocks = [[0, 1], [2, 3], [4]]
    k_blocks = [[0, 1], [2, 3], [4]]
    
    # Pre-compute Strassen factors
    sU, sV, sW = _strassen_2x2x2()
    
    # Calculate total rank
    total_rank = 0
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                ni = len(ib)
                nj = len(jb)
                nk = len(kb)
                if ni == 2 and nj == 2 and nk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * nj * nk
    
    # Initialize matrices
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                ni = len(ib)
                nj = len(jb)
                nk = len(kb)
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Use Strassen
                    for c in range(7):
                        for li in range(2):
                            for lj in range(2):
                                gi = ib[li]
                                gj = jb[lj]
                                U[gi * m + gj][col + c] += sU[li * 2 + lj][c]
                        for lj in range(2):
                            for lk in range(2):
                                gj = jb[lj]
                                gk = kb[lk]
                                V[gj * p + gk][col + c] += sV[lj * 2 + lk][c]
                        for lk in range(2):
                            for li in range(2):
                                gk = kb[lk]
                                gi = ib[li]
                                W[gk * n + gi][col + c] += sW[lk * 2 + li][c]
                    col += 7
                else:
                    # Schoolbook
                    for li in range(ni):
                        for lj in range(nj):
                            for lk in range(nk):
                                gi = ib[li]
                                gj = jb[lj]
                                gk = kb[lk]
                                U[gi * m + gj][col] = 1.0
                                V[gj * p + gk][col] = 1.0
                                W[gk * n + gi][col] = 1.0
                                col += 1
    
    assert col == total_rank
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
