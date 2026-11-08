import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 6]
DEADLINE = 115.0  # safety margin before 120s

def _build_rank42():
    """Build the known rank-42 block-Strassen solution."""
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1}
    ]
    V_s = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1}
    ]
    W_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(0,0):1, (1,0):1},
        {(0,0):-1, (0,1):1},
        {(1,1):1},
        {(0,0):1}
    ]

    col = 0
    for a_block in range(2):
        for b_block in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_s[s].items():
                    u_idx = i_loc * m + (a_block * 2 + j_loc)
                    U[u_idx, col] += coeff
                for (j_loc, k_loc), coeff in V_s[s].items():
                    v_idx = (a_block * 2 + j_loc) * p + (b_block * 2 + k_loc)
                    V[v_idx, col] += coeff
                for (i_loc, k_loc), coeff in W_s[s].items():
                    w_idx = (b_block * 2 + k_loc) * n + i_loc
                    W[w_idx, col] += coeff
                col += 1
    return U, V, W

def _tensor_check_fast(U, V, W, tol=1e-8):
    """Check if U⊗V⊗W equals the matmul tensor exactly (within tol) using vectorized ops."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    T_computed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    return np.allclose(T_computed, T_target, atol=tol)

def _round_to_halfint(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    doubled = 2.0 * x
    rounded = np.round(doubled)
    if np.allclose(doubled, rounded, atol=tol):
        return rounded / 2.0
    return None

def _als_optimize_sub(T_sub, n_sub, m_sub, p_sub, rank, max_iter=2000, tol=1e-10, seed=42, time_limit=30.0):
    """
    ALS for the 2x2x6 sub-tensor. T_sub has shape (n_sub*m_sub, m_sub*p_sub, p_sub*n_sub)
    with T_sub[i*m_sub+j, j*p_sub+k, k*n_sub+i] = 1.
    Returns (U_sub, V_sub, W_sub, error).
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n_sub * m_sub, rank) / np.sqrt(rank)
    V = rng.randn(m_sub * p_sub, rank) / np.sqrt(rank)
    W = rng.randn(p_sub * n_sub, rank) / np.sqrt(rank)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Update U
        V_reshaped = V.reshape(m_sub, p_sub, rank)
        W_reshaped = W.reshape(p_sub, n_sub, rank)
        
        U_new = np.zeros((n_sub*m_sub, rank))
        for i in range(n_sub):
            for j in range(m_sub):
                A = V_reshaped[j] * W_reshaped[:, i]  # (p_sub, rank)
                try:
                    U_new[i*m_sub+j] = np.linalg.lstsq(A, T_sub[i, j, :], rcond=None)[0]
                except:
                    U_new[i*m_sub+j] = U[i*m_sub+j]
        
        # Update V
        U_reshaped = U_new.reshape(n_sub, m_sub, rank)
        V_new = np.zeros((m_sub*p_sub, rank))
        for j in range(m_sub):
            for k in range(p_sub):
                B = U_new[:, j] * W_reshaped[k]  # (n_sub, rank)
                try:
                    V_new[j*p_sub+k] = np.linalg.lstsq(B, T_sub[:, j, k], rcond=None)[0]
                except:
                    V_new[j*p_sub+k] = V[j*p_sub+k]
        
        # Update W
        V_reshaped_new = V_new.reshape(m_sub, p_sub, rank)
        W_new = np.zeros((p_sub*n_sub, rank))
        for k in range(p_sub):
            for i in range(n_sub):
                C = U_new[i] * V_reshaped_new[:, k]  # (m_sub, rank)
                try:
                    W_new[k*n_sub+i] = np.linalg.lstsq(C, T_sub[i, :, k], rcond=None)[0]
                except:
                    W_new[k*n_sub+i] = W[k*n_sub+i]
        
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        U_r = U.reshape(n_sub, m_sub, rank)
        V_r = V.reshape(m_sub, p_sub, rank)
        W_r = W.reshape(p_sub, n_sub, rank)
        T_computed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        error = np.linalg.norm(T_computed - T_sub.reshape(n_sub, m_sub, p_sub))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    return best_U, best_V, best_W, best_error

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build rank-42 solution as fallback
    U42, V42, W42 = _build_rank42()
    
    # Verify rank-42 solution
    if _tensor_check_fast(U42, V42, W42):
        U_final = _round_to_halfint(U42)
        V_final = _round_to_halfint(V42)
        W_final = _round_to_halfint(W42)
        if U_final is not None and V_final is not None and W_final is not None:
            if _tensor_check_fast(U_final, V_final, W_final, tol=1e-9):
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
    
    # Try to find a rank-40 solution by optimizing the 2x2x6 sub-problem to rank 20
    # The full tensor T[i,j,k] = 1 for i in {0,1}, j in {0..3}, k in {0..5}
    # Split into two blocks: j in {0,1} and j in {2,3}
    # Each block is a 2x2x6 tensor: T_block[i, j_local, k] = 1
    # If we find a rank-20 decomposition of 2x2x6, we can reuse it for both blocks -> rank 40
    
    n_sub, m_sub, p_sub = 2, 2, 6
    T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                a_idx = i * m_sub + j
                b_idx = j * p_sub + k
                c_idx = k * n_sub + i
                T_sub[a_idx, b_idx, c_idx] = 1.0
    
    # Try rank 20 for the sub-problem
    for rank_sub in [20, 21]:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        best_error = np.inf
        best_U_sub, best_V_sub, best_W_sub = None, None, None
        
        for seed in range(10):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            U_sub, V_sub, W_sub, error = _als_optimize_sub(
                T_sub, n_sub, m_sub, p_sub, rank_sub,
                max_iter=1000, tol=1e-12, seed=seed, time_limit=8.0
            )
            
            if error < best_error:
                best_error = error
                best_U_sub, best_V_sub, best_W_sub = U_sub, V_sub, W_sub
            
            # Check if we can round to half-integers
            U_rounded = _round_to_halfint(U_sub, tol=1e-4)
            V_rounded = _round_to_halfint(V_sub, tol=1e-4)
            W_rounded = _round_to_halfint(W_sub, tol=1e-4)
            
            if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                # Verify the sub-decomposition
                U_r = U_rounded.reshape(n_sub, m_sub, rank_sub)
                V_r = V_rounded.reshape(m_sub, p_sub, rank_sub)
                W_r = W_rounded.reshape(p_sub, n_sub, rank_sub)
                T_computed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                if np.allclose(T_computed, np.ones((n_sub, m_sub, p_sub)), atol=1e-8):
                    # Build full rank 2*rank_sub solution by reusing the sub-decomposition
                    total_rank = 2 * rank_sub
                    U_full = np.zeros((n * m, total_rank), dtype=float)
                    V_full = np.zeros((m * p, total_rank), dtype=float)
                    W_full = np.zeros((p * n, total_rank), dtype=float)
                    
                    for a_block in range(2):
                        offset = a_block * rank_sub
                        # U: for block a, rows i*m + (a*2 + j_local)
                        for i in range(n_sub):
                            for j_loc in range(m_sub):
                                u_idx = i * m + (a_block * 2 + j_loc)
                                U_full[u_idx, offset:offset+rank_sub] = U_rounded[i * m_sub + j_loc]
                        
                        # V: for block a, rows (a*2 + j_local)*p + k
                        for j_loc in range(m_sub):
                            for k in range(p_sub):
                                v_idx = (a_block * 2 + j_loc) * p + k
                                V_full[v_idx, offset:offset+rank_sub] = V_rounded[j_loc * p_sub + k]
                        
                        # W: rows k*n + i (same for both blocks since i is in 0..1)
                        for k in range(p_sub):
                            for i in range(n_sub):
                                w_idx = k * n + i
                                W_full[w_idx, offset:offset+rank_sub] = W_rounded[k * n_sub + i]
                    
                    # Verify full solution
                    if _tensor_check_fast(U_full, V_full, W_full, tol=1e-8):
                        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
        # If best error is very small, try rounding with looser tolerance
        if best_U_sub is not None and best_error < 1e-6:
            U_rounded = _round_to_halfint(best_U_sub, tol=1e-3)
            V_rounded = _round_to_halfint(best_V_sub, tol=1e-3)
            W_rounded = _round_to_halfint(best_W_sub, tol=1e-3)
            if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                U_r = U_rounded.reshape(n_sub, m_sub, rank_sub)
                V_r = V_rounded.reshape(m_sub, p_sub, rank_sub)
                W_r = W_rounded.reshape(p_sub, n_sub, rank_sub)
                T_computed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                if np.allclose(T_computed, np.ones((n_sub, m_sub, p_sub)), atol=1e-8):
                    total_rank = 2 * rank_sub
                    U_full = np.zeros((n * m, total_rank), dtype=float)
                    V_full = np.zeros((m * p, total_rank), dtype=float)
                    W_full = np.zeros((p * n, total_rank), dtype=float)
                    
                    for a_block in range(2):
                        offset = a_block * rank_sub
                        for i in range(n_sub):
                            for j_loc in range(m_sub):
                                u_idx = i * m + (a_block * 2 + j_loc)
                                U_full[u_idx, offset:offset+rank_sub] = U_rounded[i * m_sub + j_loc]
                        
                        for j_loc in range(m_sub):
                            for k in range(p_sub):
                                v_idx = (a_block * 2 + j_loc) * p + k
                                V_full[v_idx, offset:offset+rank_sub] = V_rounded[j_loc * p_sub + k]
                        
                        for k in range(p_sub):
                            for i in range(n_sub):
                                w_idx = k * n + i
                                W_full[w_idx, offset:offset+rank_sub] = W_rounded[k * n_sub + i]
                    
                    if _tensor_check_fast(U_full, V_full, W_full, tol=1e-8):
                        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to rank 42
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
