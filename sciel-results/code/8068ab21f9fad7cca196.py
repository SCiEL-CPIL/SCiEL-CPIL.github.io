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

def _reduce_rank_svd(U, V, W, deadline):
    """
    Use SVD-based analysis to find linear dependencies and reduce rank.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # Precompute tensor for faster checks
    # We'll use a flattened representation for speed
    T_flat = T.flatten()
    total_elems = T_flat.shape[0]
    
    def _check_validity(U_curr, V_curr, W_curr):
        """Fast check if decomposition is valid using einsum."""
        T_recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
        return np.allclose(T_recon.flatten(), T_flat, atol=1e-10)

    # Greedy removal: try removing each column, keep if valid
    # We do this iteratively until no more removals are possible
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
        if best_rank <= 1:
            break

        current_rank = best_U.shape[1]
        removed = False

        for r in range(current_rank):
            if time.time() > deadline - 5:
                break

            # Create copies with column r removed
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

    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    return None

def _try_cross_block_optimization(deadline):
    """
    Try different block partitionings of the 11 columns to find lower rank.
    """
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = 44  # schoolbook

    # Try: 4 Strassen (2x2) + 1 Strassen (2x2) + 1 schoolbook (2x1) = 4*7+7+4 = 39
    # Try: 3 Strassen (2x2) + 1 Strassen (2x2) + 2 schoolbook (2x1) = 3*7+7+2*4 = 37?
    # Wait, that's only 3*2+2*2+2*1 = 12 columns, but we have 11.
    # Let's try various splits.

    # Option 1: 5 blocks of 2 + 1 block of 1 (current, rank 39)
    # Option 2: 4 blocks of 2 + 1 block of 3
    #   Block of 3: 2x2x3 schoolbook = 2*2*3 = 12, or use Strassen on first 2 + schoolbook on last 1 = 7+4=11
    #   Total: 4*7 + 11 = 39
    # Option 3: 3 blocks of 2 + 1 block of 2 + 1 block of 3 = 3*7+7+11 = 39
    # Option 4: Try to use a 2x2x3 decomposition with rank < 11

    # For 2x2x3: schoolbook rank is 12. Can we do better?
    # Strassen on first 2 cols (rank 7) + schoolbook on last 1 col (rank 4) = 11
    # Is there a rank 10 or lower for 2x2x3? Unknown, but let's try.

    # Actually, let's focus on the SVD reduction of the rank-39 construction.
    # The key insight: the 5 Strassen blocks share the same A matrix indices.
    # This means U columns from different blocks may have linear dependencies.

    # Build the standard rank-39 construction
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

    if not _verify(U, V, W):
        return None

    # Apply SVD-based rank reduction
    result = _reduce_rank_svd(U, V, W, deadline)
    if result is not None:
        U_red, V_red, W_red, new_rank = result
        if _verify(U_red, V_red, W_red) and new_rank < best_rank:
            best_rank = new_rank
            best_result = (U_red, V_red, W_red)

    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill

    # Build rank-39 construction
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

    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Try to reduce rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank39

    if time.time() < deadline - 20:
        result = _reduce_rank_svd(U, V, W, deadline)
        if result is not None:
            U_red, V_red, W_red, new_rank = result
            if _verify(U_red, V_red, W_red):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = new_rank

    # Try cross-block optimization with remaining time
    if time.time() < deadline - 10:
        cross_result = _try_cross_block_optimization(deadline)
        if cross_result is not None:
            U_x, V_x, W_x = cross_result
            rank_x = U_x.shape[1]
            if _verify(U_x, V_x, W_x) and rank_x < best_rank:
                best_U, best_V, best_W = U_x, V_x, W_x
                best_rank = rank_x

    # Convert to list and return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
