"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time
import random

_START_TIME = time.time()
_TIMEOUT = 140.0  # Safety margin before 150s kill

def _remaining_time():
    return _TIMEOUT - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    T_approx = np.zeros_like(T)
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for r in range(rank):
        T_approx += U_np[:, r][:, np.newaxis, np.newaxis] * \
                    V_np[:, r][np.newaxis, :, np.newaxis] * \
                    W_np[:, r][np.newaxis, np.newaxis, :]
                    
    return np.allclose(T_approx, T)

def _construct_strassen_238():
    """
    Constructs rank 44 decomposition for 2x3x8 matrix multiplication.
    Uses 4 blocks of 2x3x2.
    Each 2x3x2 block uses Strassen (rank 7) for 2x2x2 part and Outer Product (rank 4) for 2x1x2 part.
    Total rank: 4 * (7 + 4) = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        for idx, c in zip(u_idx, u_coeff):
            U[idx][col] += c
        for idx, c in zip(v_idx, v_coeff):
            V[idx][col] += c
        for idx, c in zip(w_idx, w_coeff):
            W[idx][col] += c

    current_col = 0
    
    for k_block in range(4):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        e_idx = 0 * p + k0
        f_idx = 0 * p + k1
        g_idx = 1 * p + k0
        h_idx = 1 * p + k1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        a_idx = 0
        b_idx = 1
        c_idx = 3
        d_idx = 4
        
        # M1 = (a+d)(e+h)
        add_term(current_col, [a_idx, d_idx], [1, 1], [e_idx, h_idx], [1, 1], [w_00, w_11], [1, 1])
        current_col += 1
        
        # M2 = (c+d)e
        add_term(current_col, [c_idx, d_idx], [1, 1], [e_idx], [1], [w_10, w_11], [1, -1])
        current_col += 1
        
        # M3 = a(f-h)
        add_term(current_col, [a_idx], [1], [f_idx, h_idx], [1, -1], [w_01, w_11], [1, 1])
        current_col += 1
        
        # M4 = d(g-e)
        add_term(current_col, [d_idx], [1], [g_idx, e_idx], [1, -1], [w_00, w_10], [1, 1])
        current_col += 1
        
        # M5 = (a+b)h
        add_term(current_col, [a_idx, b_idx], [1, 1], [h_idx], [1], [w_00, w_01], [-1, 1])
        current_col += 1
        
        # M6 = (c-a)(e+f)
        add_term(current_col, [c_idx, a_idx], [1, -1], [e_idx, f_idx], [1, 1], [w_11], [1])
        current_col += 1
        
        # M7 = (b-d)(g+h)
        add_term(current_col, [b_idx, d_idx], [1, -1], [g_idx, h_idx], [1, 1], [w_00], [1])
        current_col += 1
        
        # Part 2: j = 2 (2x1x2)
        a2_idx = 0 * m + 2
        c2_idx = 1 * m + 2
        
        e2_idx = 2 * p + k0
        f2_idx = 2 * p + k1
        
        # C[0, k0] = A[0,2] * B[2, k0]
        add_term(current_col, [a2_idx], [1], [e2_idx], [1], [w_00], [1])
        current_col += 1
        
        # C[0, k1] = A[0,2] * B[2, k1]
        add_term(current_col, [a2_idx], [1], [f2_idx], [1], [w_01], [1])
        current_col += 1
        
        # C[1, k0] = A[1,2] * B[2, k0]
        add_term(current_col, [c2_idx], [1], [e2_idx], [1], [w_10], [1])
        current_col += 1
        
        # C[1, k1] = A[1,2] * B[2, k1]
        add_term(current_col, [c2_idx], [1], [f2_idx], [1], [w_11], [1])
        current_col += 1

    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    """
    Constructs rank 48 schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _search_rank(n, m, p, target_rank, max_iters=50, restarts=5):
    """
    Attempts to find a rank `target_rank` decomposition using ALS.
    Returns None if not found.
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    T_flat_u = T.reshape(n * m, -1)
    T_flat_v = T.transpose(1, 0, 2).reshape(m * p, -1)
    T_flat_w = T.transpose(2, 0, 1).reshape(p * n, -1)

    for _ in range(restarts):
        if _remaining_time() < 10:
            break
            
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U, axis=0)
        V /= np.linalg.norm(V, axis=0)
        W /= np.linalg.norm(W, axis=0)

        for it in range(max_iters):
            if _remaining_time() < 5:
                break
                
            # Update U
            # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
            # B[v*pn+w, r] = V[v, r] W[w, r]
            # T_flat_u = U @ B.T
            try:
                # V: (mp, R), W: (pn, R)
                # prod: (mp, R, pn) -> transpose to (mp, pn, R) -> reshape (mp*pn, R)
                B = (V[:, :, np.newaxis] * W[np.newaxis, :, :]).transpose(0, 2, 1).reshape(-1, target_rank)
                U_new, _, _, _ = np.linalg.lstsq(B.T, T_flat_u.T, rcond=None)
                U = U_new.T
            except:
                continue
                
            # Update V
            # T[u, v, w] = sum_r V[v, r] U[u, r] W[w, r]
            # C[u*pn+w, r] = U[u, r] W[w, r]
            # T_flat_v = V @ C.T
            try:
                # U: (nm, R), W: (pn, R)
                # prod: (nm, R, pn) -> transpose to (nm, pn, R) -> reshape (nm*pn, R)
                C = (U[:, :, np.newaxis] * W[np.newaxis, :, :]).transpose(0, 2, 1).reshape(-1, target_rank)
                V_new, _, _, _ = np.linalg.lstsq(C.T, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                continue

            # Update W
            # T[u, v, w] = sum_r W[w, r] U[u, r] V[v, r]
            # D[u*mp+v, r] = U[u, r] V[v, r]
            # T_flat_w = W @ D.T
            try:
                # U: (nm, R), V: (mp, R)
                # prod: (nm, R, mp) -> transpose to (nm, mp, R) -> reshape (nm*mp, R)
                D = (U[:, :, np.newaxis] * V[np.newaxis, :, :]).transpose(0, 2, 1).reshape(-1, target_rank)
                W_new, _, _, _ = np.linalg.lstsq(D.T, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                continue
            
            # Check error
            T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(T_approx - T)
            if err < 1e-8:
                # Found a solution!
                # Round to integers or half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness
                T_approx_int = np.einsum('ur,vr,wr->uvw', U_int, V_int, W_int)
                if np.allclose(T_approx_int, T):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Candidate 1: Rank 44 Strassen-based
    cand1 = _construct_strassen_238()
    if _verify_decomposition(cand1["U"], cand1["V"], cand1["W"], n, m, p):
        best_rank = 44
        best_cand = cand1
    else:
        best_rank = 48
        best_cand = _construct_schoolbook()

    # Try to find lower rank
    for target_rank in [43, 42, 41, 40, 39]:
        if _remaining_time() < 10:
            break
        if target_rank >= best_rank:
            continue
            
        print(f"Searching for rank {target_rank}...")
        res = _search_rank(n, m, p, target_rank, max_iters=20, restarts=3)
        if res:
            if _verify_decomposition(res["U"], res["V"], res["W"], n, m, p):
                best_rank = target_rank
                best_cand = res
                print(f"Found rank {target_rank}!")
                # We can stop here if we found a very low rank, or continue to try lower
                # Given time constraints, breaking is safer
                break

    return best_cand
