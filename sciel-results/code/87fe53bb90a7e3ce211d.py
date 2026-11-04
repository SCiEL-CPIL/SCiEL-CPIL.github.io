import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    # Build the known rank-28 Strassen solution
    U28, V28, W28 = _build_strassen_28(n, m, p)

    # Build the multiplication tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j       # index in U
                b = j * p + k       # index in V
                c = k * n + i       # index in W (trace convention)
                T[a, b, c] = 1.0

    # Verify rank-28 solution
    U_arr = np.array(U28, dtype=float)
    V_arr = np.array(V28, dtype=float)
    W_arr = np.array(W28, dtype=float)
    
    # Convert to numpy for fast verification
    max_err = _verify_decomp_fast(U_arr, V_arr, W_arr, T)
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        U32, V32, W32 = _build_schoolbook_32(n, m, p)
        U32_arr = np.array(U32, dtype=float)
        V32_arr = np.array(V32, dtype=float)
        W32_arr = np.array(W32, dtype=float)
        return {"U": U32, "V": V32, "W": W32}

    best_rank = 28
    best_U, best_V, best_W = U_arr, V_arr, W_arr

    # Try to find rank-27 by checking linear dependencies
    if time.time() < deadline - 20:
        result = _search_rank_27(U_arr, V_arr, W_arr, T, deadline)
        if result is not None:
            U, V, W = result
            rank_found = U.shape[1]
            # Verify the new solution
            if _verify_decomp_fast(U, V, W, T) < 1e-8:
                best_rank = rank_found
                best_U, best_V, best_W = U, V, W

    # Convert back to lists for output
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _build_strassen_28(n, m, p):
    """
    Builds a rank-28 decomposition for 2x4 * 4x4 -> 2x4.
    Structure:
    A (2x4) is split into A_L (2x2) and A_R (2x2).
    B (4x4) is split into B_TL, B_TR, B_BL, B_BR (all 2x2).
    C (2x4) is composed of C_L (2x2) and C_R (2x2).
    C_L = A_L B_TL + A_R B_BL
    C_R = A_L B_TR + A_R B_BR
    
    Each sum of two 2x2 products requires 2 * Rank(M2) = 14 multiplications.
    Total 28.
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms for 2x2 matrix multiplication
    # Local indices 0,1,2,3 correspond to:
    # 0: TL, 1: TR, 2: BL, 3: BR
    # Or row-major: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1: (X11+X22)(Y11+Y22) -> C11, C22
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),        # M2: (X21+X22)Y11 -> C21, C22
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),        # M3: X11(Y21-Y11) -> C11, C12
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),        # M4: X22(Y21-Y11) -> C11, C21
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),        # M5: (X11+X12)Y22 -> C11, C12
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),        # M6: (X21-X11)(Y11+Y12) -> C22
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})         # M7: (X12-X22)(Y21+Y22) -> C11
    ]

    # Define the 4 block multiplications needed:
    # 1. A_L * B_TL
    # 2. A_R * B_BL
    # 3. A_L * B_TR
    # 4. A_R * B_BR
    
    # Block definitions: (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
    # A is 2x4. A_L is cols 0-1. A_R is cols 2-3.
    # B is 4x4. B_TL is rows 0-1, cols 0-1. B_BL is rows 2-3, cols 0-1.
    # B_TR is rows 0-1, cols 2-3. B_BR is rows 2-3, cols 2-3.
    # C is 2x4. C_L is cols 0-1. C_R is cols 2-3.
    
    blocks = [
        # (A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off)
        (0, 0, 0, 0, 0, 0),  # A_L * B_TL -> C_L (part 1)
        (0, 2, 2, 0, 0, 0),  # A_R * B_BL -> C_L (part 2)
        (0, 0, 0, 2, 0, 2),  # A_L * B_TR -> C_R (part 1)
        (0, 2, 2, 2, 0, 2),  # A_R * B_BR -> C_R (part 2)
    ]

    col_idx = 0
    for a_r_off, a_c_off, b_r_off, b_c_off, c_r_off, c_c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Map local 2x2 indices to global indices
            # Local 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            for local_idx, val in u_coeffs.items():
                lr, lc = divmod(local_idx, 2)
                global_i = a_r_off + lr
                global_j = a_c_off + lc
                g_idx = global_i * m + global_j
                U[g_idx][col_idx] = val
                
            for local_idx, val in v_coeffs.items():
                lr, lc = divmod(local_idx, 2)
                global_j = b_r_off + lr
                global_k = b_c_off + lc
                g_idx = global_j * p + global_k
                V[g_idx][col_idx] = val
                
            for local_idx, val in w_coeffs.items():
                lr, lc = divmod(local_idx, 2)
                global_i = c_r_off + lr
                global_k = c_c_off + lc
                # W index: k*n + i
                g_idx = global_k * n + global_i
                W[g_idx][col_idx] = val
                
            col_idx += 1

    return U, V, W

def _build_schoolbook_32(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1

    return U, V, W

def _verify_decomp_fast(U_arr, V_arr, W_arr, T):
    """
    Fast verification using tensor reconstruction.
    T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    """
    nm, mp, pn = T.shape
    rank = U_arr.shape[1]
    S = np.zeros((nm, mp, pn), dtype=float)
    
    # Use einsum for efficiency
    # S = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    # This creates a large intermediate, but nm*mp*pn is only 1024.
    try:
        S = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    except Exception:
        # Fallback to loop if einsum fails (unlikely)
        for r in range(rank):
            S += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
            
    return np.max(np.abs(S - T))

def _search_rank_27(U_arr, V_arr, W_arr, T, deadline):
    """
    Search for rank-27 by checking if the 28 Strassen vectors are linearly dependent.
    """
    rank = U_arr.shape[1]
    nm, mp, pn = T.shape
    total_dim = nm * mp * pn

    # Build M matrix: M[abc, r] = U[a,r]*V[b,r]*W[c,r]
    M = np.zeros((total_dim, rank), dtype=float)
    
    # Vectorized construction
    # U: (8, 28), V: (16, 28), W: (8, 28)
    # We want M: (1024, 28)
    # M[:, r] is the flattened tensor T_r = U[:,r] V[:,r] W[:,r]
    
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten()

    # Check rank of M
    # If rank(M) < 28, there is a linear dependency.
    try:
        s = np.linalg.svd(M, full_matrices=False)
        effective_rank = np.sum(s > 1e-8)
        
        if effective_rank < rank:
            # Find a basis
            # Use QR with pivoting on M.T to find independent columns
            Q, R, piv = np.linalg.qr(M.T, mode='complete', pivoting=True)
            # The first 'effective_rank' columns in M.T (which are rows in M) corresponding to piv are independent?
            # Actually, QR with pivoting on M.T (28x1024) gives pivoting of the 28 columns.
            # piv[0] is the index of the column of M.T with largest norm (after orthogonalization).
            # We want to keep 'effective_rank' columns.
            selected = piv[:effective_rank]
            
            U_new = U_arr[:, selected]
            V_new = V_arr[:, selected]
            W_new = W_arr[:, selected]
            
            # Verify the reduced solution
            if _verify_decomp_fast(U_new, V_new, W_new, T) < 1e-8:
                return U_new, V_new, W_new
            else:
                # If verification fails, it might be due to numerical noise in SVD threshold.
                # Try to remove the column with smallest singular value explicitly.
                min_idx = np.argmin(s)
                mask = [i for i in range(rank) if i != min_idx]
                U_try = U_arr[:, mask]
                V_try = V_arr[:, mask]
                W_try = W_arr[:, mask]
                if _verify_decomp_fast(U_try, V_try, W_try, T) < 1e-8:
                    return U_try, V_try, W_try
                
    except np.linalg.LinAlgError:
        pass

    # If simple dependency check fails, try random projections?
    # Given the strict 120s limit and the high likelihood that rank is 28,
    # we skip expensive random searches to ensure we return the valid rank-28 solution in time.
    
    return None

