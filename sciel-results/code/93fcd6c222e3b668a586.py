"""Matrix multiplication 4x4x6 via Strassen tiling with border-rank optimization."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

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

def _try_border_rank_optimization(U0, V0, W0, n, m, p, target_rank, deadline):
    """
    Try to find a rank-`target_rank` decomposition by optimizing from a perturbed 
    rank-(target_rank+1) initialization. Returns (U, V, W, success) or None.
    """
    U = U0.astype(np.float64).copy()
    V = V0.astype(np.float64).copy()
    W = W0.astype(np.float64).copy()
    
    # Start from rank 77, try to reduce to target_rank by dropping columns
    # and adjusting the rest via gradient descent
    rank = U.shape[1]
    
    if rank <= target_rank:
        return None
    
    # We'll work with a subset of columns. Pick which columns to keep.
    # Strategy: try dropping each column one at a time and see if we can recover
    # by optimizing the remaining columns.
    
    # For efficiency, let's try a few different column-drop strategies
    # and pick the best one.
    
    n_keep = target_rank
    n_drop = rank - n_keep
    
    # Try dropping the last n_drop columns first (simplest)
    # Then try other patterns if that fails
    
    U_keep = U[:, :n_keep].copy()
    V_keep = V[:, :n_keep].copy()
    W_keep = W[:, :n_keep].copy()
    
    # Add small perturbation to help escape potential saddle points
    np.random.seed(42)
    U_keep += 1e-4 * np.random.randn(*U_keep.shape)
    V_keep += 1e-4 * np.random.randn(*V_keep.shape)
    W_keep += 1e-4 * np.random.randn(*W_keep.shape)
    
    # Gradient descent on the tensor residual
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Want T = 1 everywhere
    # Loss = 0.5 * sum_{i,j,k} (T[i,j,k] - 1)^2
    
    lr = 1e-3
    max_iter = 5000
    tol = 1e-8
    
    Ur = U_keep.reshape(n, m, n_keep)
    Vr = V_keep.reshape(m, p, n_keep)
    Wr = W_keep.reshape(p, n, n_keep)
    
    prev_loss = float('inf')
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute tensor
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        R = T - 1.0
        loss = 0.5 * np.sum(R**2)
        
        if loss < tol:
            break
        
        # Gradient computation
        # dL/dU[i,j,c] = sum_k R[i,j,k] * V[j,k,c] * W[k,i,c]
        # dL/dV[j,k,c] = sum_i R[i,j,k] * U[i,j,c] * W[k,i,c]
        # dL/dW[k,i,c] = sum_j R[i,j,k] * U[i,j,c] * V[j,k,c]
        
        dU = np.einsum('ijk,jkc,kic->ijc', R, Vr, Wr)
        dV = np.einsum('ijk,ijc,kic->jkc', R, Ur, Wr)
        dW = np.einsum('ijk,ijc,jkc->kic', R, Ur, Vr)
        
        # Adaptive learning rate: scale down if loss increased
        if loss > prev_loss:
            lr *= 0.5
        else:
            lr = min(lr * 1.05, 1e-2)
        prev_loss = loss
        
        # Update
        Ur -= lr * dU
        Vr -= lr * dV
        Wr -= lr * dW
    
    # Final check
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    
    if err < 1e-6:
        return Ur.reshape(-1, n_keep), Vr.reshape(-1, n_keep), Wr.reshape(-1, n_keep), True
    
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
    
    # Try border-rank optimization to find rank-76 solution
    if time.time() < deadline - 20.0:
        result = _try_border_rank_optimization(U77, V77, W77, n, m, p, 76, deadline - 5.0)
        if result is not None:
            U_opt, V_opt, W_opt, success = result
            if success:
                # Verify once more
                err_opt = _verify(U_opt, V_opt, W_opt, n, m, p, 76)
                if err_opt < 1e-6:
                    return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    # Return rank-77 solution
    return {"U": U77.tolist(), "V": V77.tolist(), "W": W77.tolist()}
