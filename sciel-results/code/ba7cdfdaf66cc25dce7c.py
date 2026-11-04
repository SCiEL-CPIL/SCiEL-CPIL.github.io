"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]
_DEADLINE = time.time() + 140  # 10s safety margin

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook decomposition of (n,m,p) matrix mult."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return U, V, W for Strassen decomposition of 2x2x2 matrix mult. Rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+b)(e+f)
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10
    W[3, 1] = -1  # C11
    
    # M3 = a(g+h)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = 1
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = d(f-h)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (a+d)(e+h)
    U[0, 4] = 1; U[3, 4] = 1
    V[0, 4] = 1; V[3, 4] = 1
    W[0, 4] = -1  # C00
    W[1, 4] = 1  # C01
    
    # M6 = (b-c)(g+h)
    U[1, 5] = 1; U[2, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1  # C11
    
    # M7 = (a-c)(e+f)
    U[0, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1  # C00
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast verification using vectorized tensor operations."""
    rank = U.shape[1]
    U_3d = U.T.reshape(rank, n, m)
    V_3d = V.T.reshape(rank, m, p)
    W_3d = W.T.reshape(rank, p, n)
    S = np.einsum('sij,sjk,ski->ijk', U_3d, V_3d, W_3d)
    return np.allclose(S, 1.0, atol=tol)

def _combine_decompositions(decomps, n, m, p):
    """Combine multiple decompositions that cover disjoint sub-problems."""
    total_rank = sum(d[0].shape[1] for d in decomps)
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col = 0
    for U, V, W, i_range, j_range, k_range in decomps:
        r = U.shape[1]
        for idx_u, (ii, jj) in enumerate(product(i_range, j_range)):
            U_full[ii * m + jj, col:col+r] = U[idx_u, :]
        for idx_v, (jj, kk) in enumerate(product(j_range, k_range)):
            V_full[jj * p + kk, col:col+r] = V[idx_v, :]
        for idx_w, (kk, ii) in enumerate(product(k_range, i_range)):
            W_full[kk * n + ii, col:col+r] = W[idx_w, :]
        col += r
    
    return U_full, V_full, W_full

def _block_decomp(n, m, p, i_range, j_range, k_range):
    """Get schoolbook decomposition for a sub-block."""
    ni = len(i_range)
    nj = len(j_range)
    nk = len(k_range)
    U, V, W = _schoolbook(ni, nj, nk)
    return U, V, W, i_range, j_range, k_range

def _strassen_decomp(i_range, j_range, k_range):
    """Get Strassen decomposition for a 2x2x2 sub-block."""
    U, V, W = _strassen_2x2x2()
    return U, V, W, i_range, j_range, k_range

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: Split 5 into 2+3
    # A (2x5) = [A1 (2x2) | A2 (2x3)]
    # B (5x5) = [[B11 (2x2), B12 (2x3)], [B21 (3x2), B22 (3x3)]]
    # C (2x5) = [C1 (2x2) | C2 (2x3)]
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # Sub-problems:
    # 1. (2,2,2): A1*B11 -> C1_left  [i:0-1, j:0-1, k:0-1]
    # 2. (2,3,2): A2*B21 -> C1_right [i:0-1, j:2-4, k:0-1]
    # 3. (2,2,3): A1*B12 -> C2_left  [i:0-1, j:0-1, k:2-4]
    # 4. (2,3,3): A2*B22 -> C2_right [i:0-1, j:2-4, k:2-4]
    
    # Use Strassen for (2,2,2), schoolbook for others
    # Rank: 7 + 12 + 12 + 18 = 49
    
    decomps = []
    
    # 1. (2,2,2) with Strassen
    i1, j1, k1 = [0,1], [0,1], [0,1]
    decomps.append(_strassen_decomp(i1, j1, k1))
    
    # 2. (2,3,2) schoolbook
    i2, j2, k2 = [0,1], [2,3,4], [0,1]
    decomps.append(_block_decomp(n, m, p, i2, j2, k2))
    
    # 3. (2,2,3) schoolbook
    i3, j3, k3 = [0,1], [0,1], [2,3,4]
    decomps.append(_block_decomp(n, m, p, i3, j3, k3))
    
    # 4. (2,3,3) schoolbook
    i4, j4, k4 = [0,1], [2,3,4], [2,3,4]
    decomps.append(_block_decomp(n, m, p, i4, j4, k4))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Verify using fast method
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
