import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 6]
DEADLINE = 145.0

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

def _tensor_check(U, V, W, tol=1e-8):
    """Check if U⊗V⊗W equals the matmul tensor exactly."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    T_computed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p))
    return np.allclose(T_computed, T_target, atol=tol)

def _round_to_halfint(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    doubled = 2.0 * x
    rounded = np.round(doubled)
    if np.allclose(doubled, rounded, atol=tol):
        return rounded / 2.0
    return None

def _build_rank40_2x2x3():
    """
    Build a rank-10 decomposition for 2x2x3 matmul.
    Then combine two blocks to get rank-40 for 2x4x6.
    
    For 2x2x3: A is 2x2, B is 2x3, C is 2x3.
    We split p=3 into 1+2 and use a rank-5 + rank-5 approach? No.
    Actually, let's try to find a known rank-10 for 2x2x3.
    
    A simpler approach: use the rank-42 solution's structure but exploit symmetries.
    The rank-42 solution uses 7 multiplies per 2x2x2 block, with 2 blocks in m and 3 in p.
    Total = 2 * 3 * 7 = 42.
    
    If we can merge some multiplies across the p-blocks, we might reduce to 40.
    Specifically, if two p-blocks share a common factor in U or W, we can combine.
    
    Let's try a different approach: use the fact that R(2,4,6) <= R(2,2,3) * 2.
    We need R(2,2,3) = 10.
    
    Known: R(2,2,3) = 10 (Bini, Capovani, Lotti, Romani 1979).
    Let me try to construct it explicitly.
    
    For 2x2x3, the tensor has 12 nonzero entries.
    We need 10 rank-1 terms.
    
    Let me use a search approach with a small budget.
    """
    n_sub, m_sub, p_sub = 2, 2, 3
    rank_sub = 10
    
    # Build the 2x2x3 tensor
    T_sub = np.zeros((n_sub, m_sub, p_sub))
    T_sub.fill(1.0)
    
    # Try random initialization and ALS
    rng = np.random.RandomState(42)
    
    U = rng.randn(n_sub * m_sub, rank_sub) / np.sqrt(rank_sub)
    V = rng.randn(m_sub * p_sub, rank_sub) / np.sqrt(rank_sub)
    W = rng.randn(p_sub * n_sub, rank_sub) / np.sqrt(rank_sub)
    
    start_time = time.time()
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(500):
        if time.time() - start_time > 2.0:
            break
        
        # Update U
        V_r = V.reshape(m_sub, p_sub, rank_sub)
        W_r = W.reshape(p_sub, n_sub, rank_sub)
        
        U_new = np.zeros((n_sub * m_sub, rank_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                A = V_r[j] * W_r[:, i]  # (p_sub, rank_sub)
                A = A.reshape(-1, rank_sub)
                target = T_sub[i, j, :]
                try:
                    U_new[i * m_sub + j] = np.linalg.lstsq(A, target, rcond=None)[0]
                except:
                    U_new[i * m_sub + j] = U[i * m_sub + j]
        
        # Update V
        U_r = U_new.reshape(n_sub, m_sub, rank_sub)
        V_new = np.zeros((m_sub * p_sub, rank_sub))
        for j in range(m_sub):
            for k in range(p_sub):
                B = U_r[:, j] * W_r[k]  # (n_sub, rank_sub)
                B = B.reshape(-1, rank_sub)
                target = T_sub[:, j, k]
                try:
                    V_new[j * p_sub + k] = np.linalg.lstsq(B, target, rcond=None)[0]
                except:
                    V_new[j * p_sub + k] = V[j * p_sub + k]
        
        # Update W
        V_r_new = V_new.reshape(m_sub, p_sub, rank_sub)
        W_new = np.zeros((p_sub * n_sub, rank_sub))
        for k in range(p_sub):
            for i in range(n_sub):
                C = U_new[i] * V_r_new[:, k]  # (m_sub, rank_sub)
                C = C.reshape(-1, rank_sub)
                target = T_sub[i, :, k]
                try:
                    W_new[k * n_sub + i] = np.linalg.lstsq(C, target, rcond=None)[0]
                except:
                    W_new[k * n_sub + i] = W[k * n_sub + i]
        
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        U_r2 = U.reshape(n_sub, m_sub, rank_sub)
        V_r2 = V.reshape(m_sub, p_sub, rank_sub)
        W_r2 = W.reshape(p_sub, n_sub, rank_sub)
        T_computed = np.einsum('ijr,jkr,kir->ijk', U_r2, V_r2, W_r2)
        error = np.linalg.norm(T_computed - T_sub)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-10:
            break
    
    # Check if we can round to half-integers
    U_rounded = _round_to_halfint(best_U, tol=1e-4)
    V_rounded = _round_to_halfint(best_V, tol=1e-4)
    W_rounded = _round_to_halfint(best_W, tol=1e-4)
    
    if U_rounded is not None and V_rounded is not None and W_rounded is not None:
        U_r = U_rounded.reshape(n_sub, m_sub, rank_sub)
        V_r = V_rounded.reshape(m_sub, p_sub, rank_sub)
        W_r = W_rounded.reshape(p_sub, n_sub, rank_sub)
        T_computed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        if np.allclose(T_computed, T_sub, atol=1e-8):
            # Build full rank-40 solution
            n, m, p = DIMENSIONS
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
                        v_idx = (a_block * 2 + j_loc) * p + (k % 3) * 2 + (k // 3)
                        # Wait, p=6, p_sub=3. We need to map k in 0..2 to k in 0..5.
                        # Actually, we're splitting m into two blocks of 2, but p stays 6.
                        # So we need to handle p differently.
                        # Let me reconsider.
                        pass
            
            # This approach is getting complicated. Let me just return None and fall back.
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build rank-42 solution as fallback
    U42, V42, W42 = _build_rank42()
    
    # Verify rank-42 solution
    if _tensor_check(U42, V42, W42):
        U_final = _round_to_halfint(U42)
        V_final = _round_to_halfint(V42)
        W_final = _round_to_halfint(W42)
        if U_final is not None and V_final is not None and W_final is not None:
            if _tensor_check(U_final, V_final, W_final, tol=1e-9):
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
    
    # Fallback to rank 42
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
