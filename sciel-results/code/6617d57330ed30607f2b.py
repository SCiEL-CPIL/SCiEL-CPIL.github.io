"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time
import random

_START_TIME = time.time()
_TIMEOUT = 140.0  # Safety margin before 150s kill

def _remaining_time():
    return _TIMEOUT - (time.time() - _START_TIME)

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
        
        add_term(current_col, [a2_idx], [1], [e2_idx], [1], [w_00], [1])
        current_col += 1
        add_term(current_col, [a2_idx], [1], [f2_idx], [1], [w_01], [1])
        current_col += 1
        add_term(current_col, [c2_idx], [1], [e2_idx], [1], [w_10], [1])
        current_col += 1
        add_term(current_col, [c2_idx], [1], [f2_idx], [1], [w_11], [1])
        current_col += 1

    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    """Constructs rank 48 schoolbook decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if the tensor decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_approx = np.zeros((n * m, m * p, p * n), dtype=float)
    for r in range(rank):
        T_approx += U_np[:, r][:, np.newaxis, np.newaxis] * \
                    V_np[:, r][np.newaxis, :, np.newaxis] * \
                    W_np[:, r][np.newaxis, np.newaxis, :]
    
    return np.allclose(T_approx, T, atol=1e-9)

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using matrix multiplication trick."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    # Check against known tensor
    T_approx = np.zeros((n * m, m * p, p * n), dtype=float)
    for r in range(rank):
        T_approx += U_np[:, r][:, np.newaxis, np.newaxis] * \
                    V_np[:, r][np.newaxis, :, np.newaxis] * \
                    W_np[:, r][np.newaxis, np.newaxis, :]
    
    # Check all non-zero positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T_approx[u, v, w] - 1.0) > 1e-9:
                    return False
    # Check some zero positions (sample)
    rng = np.random.RandomState(42)
    for _ in range(1000):
        u = rng.randint(n * m)
        v = rng.randint(m * p)
        w = rng.randint(p * n)
        # Check if this is a valid (i,j,k)
        i = u // m
        j = u % m
        jj = v // p
        k = v % p
        kk = w // n
        ii = w % n
        if not (i == ii and j == jj and k == kk):
            if abs(T_approx[u, v, w]) > 1e-9:
                return False
    return True

def _random_rank_search(n, m, p, target_rank, n_trials=50, max_iter=200, seed=42):
    """
    Try to find a decomposition of given rank using random initialization + ALS-style refinement.
    Returns dict or None.
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor (flattened)
    T_flat = np.zeros(nm * mp * pn)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_flat[u * mp * pn + v * pn + w] = 1.0
    
    # Reshape to (nm, mp*pn) for ALS
    T_mat = T_flat.reshape(nm, mp * pn)
    
    best_result = None
    
    for trial in range(n_trials):
        if _remaining_time() < 10.0:
            break
            
        # Random initialization
        U = rng.randn(nm, target_rank) * 0.1
        V = rng.randn(mp, target_rank) * 0.1
        W = rng.randn(pn, target_rank) * 0.1
        
        # ALS iterations
        for it in range(max_iter):
            if _remaining_time() < 5.0:
                    break
            
            # Update U: for each u, U[u,:] = (V^T W^T T_u) / (V^T W^T V^T W)
            # T_u[v,w] = T[u, v, w]
            # sum_v,w T[u,v,w] V[v,r] W[w,r'] for all r,r'
            # = sum_r' W[:,r']^T * (T_u @ V[:,r'])
            
            # More efficient: 
            # For each u: U[u,:] = (sum_w T[u,:,w] W[w,:])^T @ V^T ... 
            # Actually: U[u,r] = sum_{v,w} T[u,v,w] V[v,r] W[w,r]
            # Let's compute: M_u = sum_w T[u,:,w] * W[w,:]  -> (mp, target_rank)
            # Then U[u,:] = M_u @ V^T  -> no, that gives (mp, mp)...
            # U[u,r] = sum_v (sum_w T[u,v,w] W[w,r]) V[v,r]
            # = (T_u @ W) @ V^T ... no
            # U[u,r] = sum_v sum_w T[u,v,w] V[v,r] W[w,r]
            # = sum_w W[w,r] * (sum_v T[u,v,w] V[v,r])
            # = sum_w W[w,r] * (T_u[:,w] dot V[:,r])
            # Let A_u = T_u @ V  -> (mp, target_rank), A_u[v,r] = sum_w T[u,v,w] V[v,r]... no
            # A_u[v,r] = sum_w T[u,v,w] * ??? 
            # T_u is (mp, pn). V is (mp, target_rank). W is (pn, target_rank).
            # U[u,r] = sum_{v,w} T[u,v,w] V[v,r] W[w,r]
            # = (T_u * (V @ W^T)^T).sum()  where T_u is (mp,pn)
            # = T_u.flatten() dot (V @ W^T).flatten()
            
            # For efficiency, precompute VW = V @ W^T -> (mp, pn)
            VW = V @ W.T  # (mp, pn)
            
            for u in range(nm):
                # U[u,r] = sum_{v,w} T[u,v,w] V[v,r] W[w,r]
                # = T[u,:].flatten() dot (V @ W.T).flatten()  -- NO, that's not right
                # U[u,r] = sum_v sum_w T[u,v,w] V[v,r] W[w,r]
                # This is elementwise: T[u] * (V[:,r] W[:,r]) summed
                # = T[u].flatten() dot np.outer(V[:,r], W[:,r]).flatten() for each r
                # = T[u].flatten() dot (V * W).flatten() ... no, V and W have different shapes
                
                # Better: U[u,r] = (T[u] * (V[:,r][:,None] * W[:,r][None,:])).sum()
                # For all r: U[u,:] = T[u].flatten() dot (V * W.T).flatten() -- still not right
                
                # Let's just do it directly but vectorized:
                # U[u,:] = np.array([T[u].flatten() @ (V[:,r] * W[:,r]).flatten() for r in range(target_rank)])
                # = T[u].flatten() @ np.einsum('vr,wr->vwr', V, W).reshape(mp*pn)
                # Precompute: outer_VW = np.einsum('vr,wr->vwr', V, W) -> (mp, pn, target_rank)
                # Then U[u,:] = T[u].reshape(mp, pn) * outer_VW, summed over v,w
                
                # Even better: U[u,:] = np.einsum('vw,vwr->r', T[u], outer_VW)
                # But we need outer_VW for each trial. Let's precompute once per iteration.
                
                pass  # We'll handle this below with precomputed outer_VW
            
            # Precompute outer_VW = einsum('vr,wr->vwr', V, W) -> (mp, pn, target_rank)
            outer_VW = np.einsum('vr,wr->vwr', V, W)  # (mp, pn, R)
            
            for u in range(nm):
                Tu = T_mat[u].reshape(mp, pn)
                # U[u,r] = sum_{v,w} Tu[v,w] * outer_VW[v,w,r]
                U[u, :] = np.einsum('vw,vwr->r', Tu, outer_VW)
            
            # Update V: V[v,r] = sum_{u,w} T[u,v,w] U[u,r] W[w,r]
            # Precompute outer_UW = einsum('ur,wr->uwr', U, W) -> (nm, pn, R)
            outer_UW = np.einsum('ur,wr->uwr', U, W)  # (nm, pn, R)
            
            for v in range(mp):
                # T[:,v,:] is (nm, pn)
                Tv = T_flat.reshape(nm, mp, pn)[:, v, :]
                V[v, :] = np.einsum('uw,uwr->r', Tv, outer_UW)
            
            # Update W: W[w,r] = sum_{u,v} T[u,v,w] U[u,r] V[v,r]
            # Precompute outer_UV = einsum('ur,vr->uvr', U, V) -> (nm, mp, R)
            outer_UV = np.einsum('ur,vr->uvr', U, V)  # (nm, mp, R)
            
            for w in range(pn):
                Tw = T_flat.reshape(nm, mp, pn)[:, :, w]
                W[w, :] = np.einsum('uv,uvr->r', Tw, outer_UV)
        
        # Check if this decomposition is exact
        if _remaining_time() < 5.0:
            break
            
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Also try rounding to nearest 0.5
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        U_list = [[float(x) for x in row] for row in U_round]
        V_list = [[float(x) for x in row] for row in V_round]
        W_list = [[float(x) for x in row] for row in W_round]
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _structured_search(n, m, p, target_rank, n_trials=20, seed=100):
    """
    Search with structured initialization: start from Strassen-like patterns.
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    T_flat = np.zeros(nm * mp * pn)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_flat[u * mp * pn + v * pn + w] = 1.0
    
    T_mat = T_flat.reshape(nm, mp * pn)
    
    for trial in range(n_trials):
        if _remaining_time() < 10.0:
            break
        
        # Initialize with random sparse structure
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        # Each column has a few non-zero entries
        for r in range(target_rank):
            # Randomly pick 2-3 entries for each factor
            n_nonzero_u = rng.randint(2, 4)
            n_nonzero_v = rng.randint(2, 4)
            n_nonzero_w = rng.randint(2, 4)
            
            u_indices = rng.choice(nm, n_nonzero_u, replace=False)
            v_indices = rng.choice(mp, n_nonzero_v, replace=False)
            w_indices = rng.choice(pn, n_nonzero_w, replace=False)
            
            for idx in u_indices:
                U[idx, r] = rng.choice([-1, 1])
            for idx in v_indices:
                V[idx, r] = rng.choice([-1, 1])
            for idx in w_indices:
                W[idx, r] = rng.choice([-1, 1])
        
        # ALS refinement
        max_iter = 100
        for it in range(max_iter):
            if _remaining_time() < 5.0:
                break
            
            outer_VW = np.einsum('vr,wr->vwr', V, W)
            for u in range(nm):
                Tu = T_mat[u].reshape(mp, pn)
                U[u, :] = np.einsum('vw,vwr->r', Tu, outer_VW)
            
            outer_UW = np.einsum('ur,wr->uwr', U, W)
            T_3d = T_flat.reshape(nm, mp, pn)
            for v in range(mp):
                V[v, :] = np.einsum('uw,uwr->r', T_3d[:, v, :], outer_UW)
            
            outer_UV = np.einsum('ur,vr->uvr', U, V)
            for w in range(pn):
                W[w, :] = np.einsum('uv,uvr->r', T_3d[:, :, w], outer_UV)
        
        if _remaining_time() < 5.0:
            break
        
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try rounding
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        U_list = [[float(x) for x in row] for row in U_round]
        V_list = [[float(x) for x in row] for row in V_round]
        W_list = [[float(x) for x in row] for row in W_round]
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = 2, 3, 8
    
    # Start with known rank-44 solution
    best = _construct_strassen_238()
    best_rank = 44
    
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        best = _construct_schoolbook()
        best_rank = 48
    
    # Try to find lower rank solutions
    for target_rank in range(43, 35, -1):
        if _remaining_time() < 15.0:
            break
        
        # Try random search
        result = _random_rank_search(n, m, p, target_rank, n_trials=30, max_iter=150, seed=target_rank)
        if result is not None:
            best = result
            best_rank = target_rank
            break
        
        if _remaining_time() < 15.0:
            break
        
        # Try structured search
        result = _structured_search(n, m, p, target_rank, n_trials=15, seed=target_rank + 1000)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    # Final verification
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
        best = _construct_schoolbook()
    
    return best
