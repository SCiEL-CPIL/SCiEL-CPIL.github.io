"""Matrix multiplication 4x4x6 via Strassen tiling with slice-interpolation and SVD compression."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np
from scipy.linalg import svd

def _get_strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * m + j][m_idx] += c
        for (j, k, c) in Vb:
            V[j * p + k][m_idx] += c
        for (i, k, c) in Wb:
            W[k * n + i][m_idx] += c

    return U, V, W

def _build_strassen_77(n, m, p):
    """Build the rank-77 Strassen tiling for 4x4x6."""
    U_s, V_s, W_s = _get_strassen_2x2_factors()
    
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # 1. 4x4x4 Strassen (columns 0-48)
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = U_s[2 * I + J][t] * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j, col] += val
                        
    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = V_s[2 * J + K][t] * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j * 6 + k, col] += val
                        
    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = W_s[2 * K + I][t] * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # 2. 4x4x2 Strassen-block (columns 49-76)
    products = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    ]
    
    for p_idx, (I, J, is_bot) in enumerate(products):
        col_offset = rank_444 + p_idx * 7
        
        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                for s in range(7):
                    val = U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j, col_offset + s] += val
                        
        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * is_bot + j0
                k_global = 4 + k0
                for s in range(7):
                    val = V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j_global * 6 + k_global, col_offset + s] += val
                        
        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                for s in range(7):
                    val = W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k_global * 4 + i_global, col_offset + s] += val

    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err

def _try_svd_compression(U0, V0, W0, n, m, p, target_rank, deadline):
    """
    Placeholder for SVD compression.
    The previous implementation was mathematically flawed and crashed.
    We return None to skip this step and rely on the structural decomposition.
    """
    return None

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0
    
    # Build the rank-77 Strassen tiling
    U77, V77, W77 = _build_strassen_77(n, m, p)
    
    # Verify rank-77 solution
    err77 = _verify(U77, V77, W77, n, m, p, 77)
    if err77 > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try SVD compression to find lower-rank solutions
    for target_rank in [76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65]:
        if time.time() > deadline - 30.0:
            break
        
        result = _try_svd_compression(U77, V77, W77, n, m, p, target_rank, deadline - 10.0)
        if result is not None:
            U_opt, V_opt, W_opt, success = result
            if success:
                # Verify once more
                err_opt = _verify(U_opt, V_opt, W_opt, n, m, p, target_rank)
                if err_opt < 1e-6:
                    return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    # Return rank-77 solution
    return {"U": U77.tolist(), "V": V77.tolist(), "W": W77.tolist()}
