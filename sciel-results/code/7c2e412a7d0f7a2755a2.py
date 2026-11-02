import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _build_strassen_block(U, V, W, r, k0, k1):
    """Add a Strassen 2x2 block for column pair (k0, k1). Returns new r."""
    v_e = k0
    v_f = k1
    v_g = 11 + k0
    v_h = 11 + k1
    w_c00 = 2 * k0
    w_c10 = 2 * k0 + 1
    w_c01 = 2 * k1
    w_c11 = 2 * k1 + 1

    # m1 = (a+d)(e+h)
    U[0, r] += 1; U[3, r] += 1
    V[v_e, r] += 1; V[v_h, r] += 1
    W[w_c00, r] += 1; W[w_c11, r] += 1
    r += 1
    # m2 = (c+d)e
    U[2, r] += 1; U[3, r] += 1
    V[v_e, r] += 1
    W[w_c10, r] += 1; W[w_c11, r] += -1
    r += 1
    # m3 = a(f-h)
    U[0, r] += 1
    V[v_f, r] += 1; V[v_h, r] += -1
    W[w_c01, r] += 1; W[w_c11, r] += 1
    r += 1
    # m4 = d(g-e)
    U[3, r] += 1
    V[v_g, r] += 1; V[v_e, r] += -1
    W[w_c00, r] += 1; W[w_c10, r] += 1
    r += 1
    # m5 = (a+b)h
    U[0, r] += 1; U[1, r] += 1
    V[v_h, r] += 1
    W[w_c00, r] += -1; W[w_c01, r] += 1
    r += 1
    # m6 = (c-a)(e+f)
    U[2, r] += 1; U[0, r] += -1
    V[v_e, r] += 1; V[v_f, r] += 1
    W[w_c11, r] += 1
    r += 1
    # m7 = (b-d)(g+h)
    U[1, r] += 1; U[3, r] += -1
    V[v_g, r] += 1; V[v_h, r] += 1
    W[w_c00, r] += 1
    r += 1
    return r

def _build_schoolbook_block(U, V, W, r, k):
    """Add a schoolbook 2x1 block for column k. Returns new r."""
    v_e = k
    v_g = 11 + k
    w_c0 = 2 * k
    w_c1 = 2 * k + 1

    U[0, r] += 1
    V[v_e, r] += 1
    W[w_c0, r] += 1
    r += 1
    U[1, r] += 1
    V[v_g, r] += 1
    W[w_c0, r] += 1
    r += 1
    U[2, r] += 1
    V[v_e, r] += 1
    W[w_c1, r] += 1
    r += 1
    U[3, r] += 1
    V[v_g, r] += 1
    W[w_c1, r] += 1
    r += 1
    return r

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-10)

def _build_rank39():
    """Build the standard rank-39 Strassen+schoolbook construction."""
    n, m, p = DIMENSIONS
    rank39 = 39
    U = np.zeros((n*m, rank39), dtype=np.float64)
    V = np.zeros((m*p, rank39), dtype=np.float64)
    W = np.zeros((p*n, rank39), dtype=np.float64)

    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        r = _build_strassen_block(U, V, W, r, k0, k1)
    r = _build_schoolbook_block(U, V, W, r, 10)
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-44 construction."""
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U = np.zeros((n*m, rank_sb))
    V = np.zeros((m*p, rank_sb))
    W = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _compute_tensor_lower_bound():
    """
    Compute the tensor rank lower bound via flattening matrix ranks.
    For a tensor T in A⊗B⊗C, the rank is at least the rank of any flattening.
    The 4x22 flattening (fixing the p*n=22 index) gives rank ≤ 4.
    We compute all flattening ranks to find the best lower bound.
    """
    n, m, p = DIMENSIONS
    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Flattening 1: reshape to (n*m, m*p*p*n) -> rank of this matrix
    # Actually, for lower bound, we look at matrix flattenings:
    # T as (n*m) x (m*p * p*n): rank gives lower bound
    # T as (m*p) x (n*m * p*n): rank gives lower bound  
    # T as (p*n) x (n*m * m*p): rank gives lower bound
    
    T1 = T.reshape(n*m, m*p * p*n)
    T2 = T.reshape(m*p, n*m * p*n)
    T3 = T.reshape(p*n, n*m * m*p)
    
    r1 = np.linalg.matrix_rank(T1)
    r2 = np.linalg.matrix_rank(T2)
    r3 = np.linalg.matrix_rank(T3)
    
    return max(r1, r2, r3)

def _reduce_rank_greedy(U, V, W, deadline):
    """
    Greedy rank reduction: iteratively try removing each column,
    keep if the decomposition remains valid.
    """
    n, m, p = DIMENSIONS
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_flat = T.flatten()
    
    def _check_validity(U_curr, V_curr, W_curr):
        T_recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
        return np.allclose(T_recon.flatten(), T_flat, atol=1e-10)
    
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        if best_rank <= 1:
            break

        current_rank = best_U.shape[1]
        removed = False

        for r in range(current_rank):
            if time.time() > deadline - 5:
                break

            U_red = np.delete(best_U, r, axis=1)
            V_red = np.delete(best_V, r, axis=1)
            W_red = np.delete(best_W, r, axis=1)

            if _check_validity(U_red, V_red, W_red):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = current_rank - 1
                removed = True
                break

        if not removed:
            break

    if best_rank < U.shape[1]:
        return best_U, best_V, best_W, best_rank
    return None

def _try_nullspace_decomposition(deadline):
    """
    Try to find a minimal-rank decomposition by analyzing the structure.
    
    Key insight: The 2x2x11 multiplication tensor has a specific structure.
    Each column k of the B matrix (size 2x11) contributes to the output.
    
    For the 2x2xk problem, the tensor rank is known:
    - 2x2x1: rank 4 (schoolbook)
    - 2x2x2: rank 7 (Strassen)
    - 2x2x3: rank ? (schoolbook 12, Strassen+schoolbook 11)
    
    Let's try to find if we can do better than 39 by combining blocks
    in non-standard ways.
    
    Strategy: Build the rank-39 decomposition, then use SVD on the
    combined factor matrices to find linear dependencies.
    """
    n, m, p = DIMENSIONS
    
    # Build rank-39
    U, V, W = _build_rank39()
    
    if not _verify(U, V, W):
        return None
    
    # Try greedy reduction
    result = _reduce_rank_greedy(U, V, W, deadline)
    if result is not None:
        U_red, V_red, W_red, new_rank = result
        if _verify(U_red, V_red, W_red):
            return U_red, V_red, W_red
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Compute lower bound for information
    lb = _compute_tensor_lower_bound()
    
    # Build rank-39 construction
    U, V, W = _build_rank39()
    
    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _build_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try greedy rank reduction
    if time.time() < deadline - 10:
        result = _try_nullspace_decomposition(deadline)
        if result is not None:
            U_x, V_x, W_x = result
            rank_x = U_x.shape[1]
            if rank_x < best_rank:
                best_U, best_V, best_W = U_x, V_x, W_x
                best_rank = rank_x
    
    # Convert to list and return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
