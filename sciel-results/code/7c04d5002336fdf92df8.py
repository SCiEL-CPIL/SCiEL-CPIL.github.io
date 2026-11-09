"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 140.0  # Safety margin before 150s kill

def _remaining_time():
    return _TIMEOUT - (time.time() - _START_TIME)

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, T=None):
    """Verify if the tensor decomposition is exact."""
    if T is None:
        T = _build_target_tensor(n, m, p)
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Vectorized: T_approx[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    # = einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    T_approx = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    return np.allclose(T_approx, T, atol=1e-10)

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

def _als_decompose_fast(T, n, m, p, rank, max_iter=200, seed=42, tol=1e-10):
    """
    Fast vectorized ALS for tensor decomposition.
    T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    Uses Kronecker products for efficient least squares.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Precompute flattened targets
    T_u = T.reshape(n * m, m * p, p * n)  # T[u, v, w]
    T_v = T.transpose(1, 0, 2).reshape(m * p, n * m, p * n)
    T_w = T.transpose(2, 0, 1).reshape(p * n, n * m, m * p)
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT - 15:
            break
        
        # Update U: for each u, solve U[u,:] from T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
        # Reshape: T_u[u] is (m*p, p*n). We need M such that M[r, (v,w)] = V[v,r]*W[w,r]
        # M = kron(W, V) reshaped... Actually: M[(v,w), r] = V[v,r]*W[w,r]
        # So M = np.einsum('vr,wr->vwr', V, W).reshape(m*p*p*n, rank)
        M_u = np.einsum('vr,wr->vwr', V, W).reshape(m * p * p * n, rank)
        try:
            U = np.linalg.lstsq(M_u.T, T_u.reshape(n * m, -1).T, rcond=None)[0].T
        except:
            pass
        
        # Update V: for each v, solve V[v,:] from T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
        M_v = np.einsum('ur,wr->uwr', U, W).reshape(n * m * p * n, rank)
        try:
            V = np.linalg.lstsq(M_v.T, T_v.reshape(m * p, -1).T, rcond=None)[0].T
        except:
            pass
        
        # Update W: for each w, solve W[w,:] from T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
        M_w = np.einsum('ur,vr->uvr', U, V).reshape(n * m * m * p, rank)
        try:
            W = np.linalg.lstsq(M_w.T, T_w.reshape(p * n, -1).T, rcond=None)[0].T
        except:
            pass
        
        # Check convergence
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        if np.max(np.abs(T_approx - T)) < tol:
            return U, V, W
    
    return None

def _try_rank_decomposition(n, m, p, max_rank, seed=42):
    """
    Try to find a rank decomposition using fast ALS with various initializations.
    """
    T = _build_target_tensor(n, m, p)
    
    for target_rank in range(max_rank, max(38, max_rank - 6), -1):
        if time.time() - _START_TIME > _TIMEOUT - 25:
            break
        
        for s in range(15):
            if time.time() - _START_TIME > _TIMEOUT - 25:
                break
            
            result = _als_decompose_fast(T, n, m, p, target_rank, max_iter=150, seed=seed + s, tol=1e-9)
            if result is not None:
                U, V, W = result
                T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
                if np.allclose(T_approx, T, atol=1e-8):
                    U_list = [[float(U[i, r]) for r in range(target_rank)] for i in range(n * m)]
                    V_list = [[float(V[i, r]) for r in range(target_rank)] for i in range(m * p)]
                    W_list = [[float(W[i, r]) for r in range(target_rank)] for i in range(p * n)]
                    return target_rank, U_list, V_list, W_list
    
    return None

def _try_structured_rank43(n, m, p):
    """
    Try to find a rank-43 decomposition by modifying the Strassen construction.
    The 2x3x8 tensor can be viewed as 4 blocks of 2x3x2.
    Each block has rank 11 (7 Strassen + 4 outer product for the 3rd column).
    Total: 44. We try to merge one rank-1 term across blocks.
    
    Alternative: Use a different block structure. The 2x3x8 = 2x(3x8) or (2x3)x8.
    Try: view as 2x3x8 with a different decomposition strategy.
    """
    if time.time() - _START_TIME > _TIMEOUT - 30:
        return None
    
    T = _build_target_tensor(n, m, p)
    
    # Strategy: Start from the rank-44 Strassen and try to merge terms
    # by finding linear dependencies. Use SVD-based approach.
    # Actually, let's try a direct ALS approach with structured init.
    
    # Initialize from the Strassen construction but with rank 43
    cand44 = _construct_strassen_238()
    U44 = np.array(cand44["U"], dtype=float)
    V44 = np.array(cand44["V"], dtype=float)
    W44 = np.array(cand44["W"], dtype=float)
    
    # Try to compress: use SVD on the "core" to find if rank 43 suffices
    # Reshape into a matrix: M[u*(m*p*p*n) + v*(p*n) + w, r] = U[u,r]*V[v,r]*W[w,r]
    # This is the "flattened" representation. If the tensor has rank R, 
    # this matrix has rank R.
    
    # Build the full matrix representation
    nU, nV, nW = n * m, m * p, p * n
    # M has shape (nU * nV * nW, rank) where M[idx, r] = U[u,r]*V[v,r]*W[w,r]
    # idx = u * nV * nW + v * nW + w
    M = np.zeros((nU * nV * nW, 44))
    for r in range(44):
        M[:, r] = np.einsum('u,v,w->uvw', U44[:, r], V44[:, r], W44[:, r]).flatten()
    
    # Check if the effective rank is <= 43
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # If the 44th singular value is very small, we can drop it
        if s[43] < 1e-10:
            # Rank is actually <= 43
            new_rank = 43
            U_new = U44[:, :43]
            V_new = V44[:, :43]
            W_new = W44[:, :43]
            # Verify
            T_approx = np.einsum('ur,vr,wr->uvw', U_new, V_new, W_new)
            if np.allclose(T_approx, T, atol=1e-9):
                U_list = [[float(U_new[i, r]) for r in range(new_rank)] for i in range(nU)]
                V_list = [[float(V_new[i, r]) for r in range(new_rank)] for i in range(nV)]
                W_list = [[float(W_new[i, r]) for r in range(new_rank)] for i in range(nW)]
                return new_rank, U_list, V_list, W_list
    except:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Candidate 1: Rank 44 Strassen-based
    cand1 = _construct_strassen_238()
    if _verify_decomposition(cand1["U"], cand1["V"], cand1["W"], n, m, p):
        # Try structured rank-43 first (fast)
        result = _try_structured_rank43(n, m, p)
        if result is not None and result[0] < 44:
            rank, U, V, W = result
            cand_better = {"U": U, "V": V, "W": W}
            if _verify_decomposition(cand_better["U"], cand_better["V"], cand_better["W"], n, m, p):
                return cand_better
        
        # Try ALS search for rank 43
        if _remaining_time() > 40:
            result = _try_rank_decomposition(n, m, p, 43)
            if result is not None and result[0] < 44:
                rank, U, V, W = result
                cand_better = {"U": U, "V": V, "W": W}
                if _verify_decomposition(cand_better["U"], cand_better["V"], cand_better["W"], n, m, p):
                    return cand_better
        
        return cand1
        
    # Candidate 2: Rank 48 Schoolbook (Fallback)
    cand2 = _construct_schoolbook()
    if _verify_decomposition(cand2["U"], cand2["V"], cand2["W"], n, m, p):
        return cand2
        
    # Should not happen, but return schoolbook regardless if verification fails due to bug
    return cand2
