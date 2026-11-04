"""Matrix multiplication 4x4x6 via Strassen tiling with rank-reduction search."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np
from scipy.optimize import minimize

def _get_strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

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
            U[i * m + j, m_idx] += c
        for (j, k, c) in Vb:
            V[j * p + k, m_idx] += c
        for (i, k, c) in Wb:
            W[k * n + i, m_idx] += c

    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err

def _build_444_part(U_s, V_s, W_s):
    """Build the 4x4x4 Strassen part (rank 49)."""
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)  # j in 0..3, k in 0..3 for 4x4x4
    W = np.zeros((16, 49), dtype=float)  # k in 0..3, i in 0..3 for 4x4x4
    
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            u1 = U_s[2 * I + J, :]
            u2 = U_s[2 * i0 + j0, :]
            outer = np.outer(u1, u2)
            U[i * 4 + j, :] = outer.flatten()
            
    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            v1 = V_s[2 * J + K, :]
            v2 = V_s[2 * j0 + k0, :]
            outer = np.outer(v1, v2)
            V[j * 4 + k, :] = outer.flatten()
            
    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            w1 = W_s[2 * K + I, :]
            w2 = W_s[2 * k0 + i0, :]
            outer = np.outer(w1, w2)
            W[k * 4 + i, :] = outer.flatten()
    
    return U, V, W

def _build_442_part(U_s, V_s, W_s):
    """Build the 4x4x2 Strassen-block part (rank 28).
    Returns U (16, 28), V (8, 28), W (8, 28).
    """
    rank_442 = 28
    U = np.zeros((16, rank_442), dtype=float)
    V = np.zeros((8, rank_442), dtype=float)   # j in 0..3, k in 4..5
    W = np.zeros((8, rank_442), dtype=float)   # k in 4..5, i in 0..3
    
    products = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    ]
    
    for p_idx, (I, J, is_bot) in enumerate(products):
        col_offset = p_idx * 7
        
        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                U[i * 4 + j, col_offset:col_offset+7] += U_s[2 * i0 + j0, :]
                
        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * is_bot + j0
                k_global = 4 + k0
                V[j_global * 6 + k_global, col_offset:col_offset+7] += V_s[2 * j0 + k0, :]
                
        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                W[k_global * 4 + i_global, col_offset:col_offset+7] += W_s[2 * k0 + i0, :]

    return U, V, W

def _assemble(U_444, V_444, W_444, U_442, V_442, W_442):
    """Assemble the full 4x4x6 decomposition.
    U: (16, 77), V: (24, 77), W: (24, 77)
    """
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((24, total_rank), dtype=float)  # j in 0..3, k in 0..5
    W = np.zeros((24, total_rank), dtype=float)  # k in 0..5, i in 0..3
    
    # 4x4x4 part: k in 0..3
    U[:, :49] = U_444
    V[:16, :49] = V_444  # j in 0..3, k in 0..3
    W[:16, :49] = W_444  # k in 0..3, i in 0..3
    
    # 4x4x2 part: k in 4..5
    U[:, 49:] = U_442
    V[16:, 49:] = V_442  # j in 0..3, k in 4..5
    W[16:, 49:] = W_442  # k in 4..5, i in 0..3
    
    return U, V, W

def _try_reduce_rank_444(U_444, V_444, W_444, target_rank, deadline):
    """Try to reduce the rank of the 4x4x4 part by zeroing columns.
    Returns (U, V, W, success) with shapes (16, target_rank), (16, target_rank), (16, target_rank).
    """
    current_rank = 49
    U = U_444.copy()
    V = V_444.copy()
    W = W_444.copy()
    
    # Try to remove columns one at a time
    for remove_idx in range(current_rank - target_rank):
        if time.time() > deadline:
            break
        
        # Try each remaining column
        best_err = float('inf')
        best_col = -1
        best_params = None
        
        for col in range(current_rank):
            if time.time() > deadline:
                break
            
            # Zero out this column
            U_new = U.copy()
            V_new = V.copy()
            W_new = W.copy()
            U_new[:, col] = 0
            V_new[:, col] = 0
            W_new[:, col] = 0
            
            # Check if we can adjust other columns to compensate
            # This is a linear system: sum_c' U_c' ⊗ V_c' ⊗ W_c' = T
            # We fix U and V, solve for W.
            # Actually, let's just check the error with this column zeroed.
            err = _verify(U_new, V_new, W_new, 4, 4, 4, current_rank)
            
            if err < best_err:
                best_err = err
                best_col = col
                best_params = (U_new, V_new, W_new)
        
        if best_col >= 0 and best_err < 1e-6:
            U, V, W = best_params
            # Remove the column
            mask = [i for i in range(current_rank) if i != best_col]
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            current_rank -= 1
        else:
            # Try a local optimization to see if we can reduce the error
            # by adjusting nearby columns
            pass
    
    return U, V, W, current_rank == target_rank

def _try_reduce_rank_442(U_442, V_442, W_442, target_rank, deadline):
    """Try to reduce the rank of the 4x4x2 part by zeroing columns.
    Returns (U, V, W, success) with shapes (16, target_rank), (8, target_rank), (8, target_rank).
    """
    current_rank = 28
    U = U_442.copy()
    V = V_442.copy()
    W = W_442.copy()
    
    for remove_idx in range(current_rank - target_rank):
        if time.time() > deadline:
            break
        
        best_err = float('inf')
        best_col = -1
        best_params = None
        
        for col in range(current_rank):
            if time.time() > deadline:
                break
            
            U_new = U.copy()
            V_new = V.copy()
            W_new = W.copy()
            U_new[:, col] = 0
            V_new[:, col] = 0
            W_new[:, col] = 0
            
            # Verify for 4x4x2: i in 0..3, j in 0..3, k in 0..1
            # U: (16, r), V: (8, r), W: (8, r)
            # T[i,j,k] = sum_c U[i*4+j,c] * V[j*2+k,c] * W[k*4+i,c]
            # But V and W are indexed differently. Let me check.
            # In the 4x4x2 part:
            # U[i*4+j, c] for i,j in 0..3
            # V[j*2+k, c] for j in 0..3, k in 0..1 (but we stored as V[j_global*6+k_global, c] with k_global=4+k0)
            # Actually, let me just use the verify function with appropriate dimensions.
            
            # For 4x4x2: n=4, m=4, p=2
            # U shape (16, r), V shape (8, r), W shape (8, r)
            # V index: j*p+k = j*2+k
            # W index: k*n+i = k*4+i
            
            err = _verify(U_new, V_new, W_new, 4, 4, 2, current_rank)
            
            if err < best_err:
                best_err = err
                best_col = col
                best_params = (U_new, V_new, W_new)
        
        if best_col >= 0 and best_err < 1e-6:
            U, V, W = best_params
            mask = [i for i in range(current_rank) if i != best_col]
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            current_rank -= 1
        else:
            pass
    
    return U, V, W, current_rank == target_rank

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    U_s, V_s, W_s = _get_strassen_2x2_factors()
    
    U_444, V_444, W_444 = _build_444_part(U_s, V_s, W_s)
    U_442, V_442, W_442 = _build_442_part(U_s, V_s, W_s)
    
    # Try to reduce rank of 4x4x4 part
    U_444_red, V_444_red, W_444_red, success_444 = _try_reduce_rank_444(U_444, V_444, W_444, 45, deadline)
    rank_444 = U_444_red.shape[1]
    
    # Try to reduce rank of 4x4x2 part
    U_442_red, V_442_red, W_442_red, success_442 = _try_reduce_rank_442(U_442, V_442, W_442, 24, deadline)
    rank_442 = U_442_red.shape[1]
    
    # Assemble
    total_rank = rank_444 + rank_442
    
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((24, total_rank), dtype=float)
    W = np.zeros((24, total_rank), dtype=float)
    
    U[:, :rank_444] = U_444_red
    V[:16, :rank_444] = V_444_red
    W[:16, :rank_444] = W_444_red
    
    U[:, rank_444:] = U_442_red
    V[16:, rank_444:] = V_442_red
    W[16:, rank_444:] = W_442_red
    
    err = _verify(U, V, W, n, m, p, total_rank)
    
    if err > 1e-6:
        # Fallback to original
        U_444, V_444, W_444 = _build_444_part(U_s, V_s, W_s)
        U_442, V_442, W_442 = _build_442_part(U_s, V_s, W_s)
        U, V, W = _assemble(U_444, V_444, W_444, U_442, V_442, W_442)
        total_rank = 77
        err = _verify(U, V, W, n, m, p, total_rank)
        
        if err > 1e-6:
            # Schoolbook fallback
            rank = n * m * p
            U_f = np.zeros((n * m, rank), dtype=float)
            V_f = np.zeros((m * p, rank), dtype=float)
            W_f = np.zeros((p * n, rank), dtype=float)
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_f[i * m + j, column] = 1
                        V_f[j * p + k, column] = 1
                        W_f[k * n + i, column] = 1
                        column += 1
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
