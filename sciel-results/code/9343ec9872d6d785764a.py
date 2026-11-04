"""Combined program: block-Strassen base (rank 84) + randomized algebraic rank-reduction search."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B.
    m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+b)h, m6=(c-a)(e+f), m7=(b-d)(g+h)
    C[0,0]=m1+m4-m5+m7, C[0,1]=m3+m5, C[1,0]=m2+m4, C[1,1]=m1-m2+m3+m6
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def schoolbook_factors(n, m, p):
    """Returns (U, V, W) for n x m x p schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    """
    Build the full factor matrices using block decomposition.
    Each (ni, mj, pk) block uses Strassen (rank 7) if all sizes are 2, else schoolbook.
    """
    n_offsets = []
    off = 0
    for s in n_sizes:
        n_offsets.append(off); off += s
    m_offsets = []
    off = 0
    for s in m_sizes:
        m_offsets.append(off); off += s
    p_offsets = []
    off = 0
    for s in p_sizes:
        p_offsets.append(off); off += s

    total_rank = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                total_rank += 7 if (ni == 2 and mj == 2 and pk == 2) else ni * mj * pk

    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)

    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                if ni == 2 and mj == 2 and pk == 2:
                    U_block, V_block, W_block = strassen_factors()
                else:
                    U_block, V_block, W_block = schoolbook_factors(ni, mj, pk)
                r = U_block.shape[1]
                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]
                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]
                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]
                col_offset += r

    return U_full, V_full, W_full, total_rank


def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all (i,j,k)."""
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)
    return bool(np.allclose(vals, 1.0, atol=atol))


def _find_half_integer_scaling(U_sub, V_sub, W_sub, u_rows, v_rows, w_rows, n, m, p, atol=1e-9):
    """
    Find half-integer scalings for columns of U_sub, V_sub, W_sub such that
    sum_r (a_r * U_sub[:,r]) * V_sub[:,r] * W_sub[:,r] = 1 for all entries.
    Returns (a_U, a_V, a_W) as arrays of half-integers, or None.
    """
    N = len(u_rows)
    cur_rank = U_sub.shape[1]
    
    # Build contribution matrix
    G = np.empty((N, cur_rank), dtype=np.float64)
    for r in range(cur_rank):
        G[:, r] = U_sub[u_rows, r] * V_sub[v_rows, r] * W_sub[w_rows, r]
    
    # Try scaling U only
    a, res, rank_sub, sv = np.linalg.lstsq(G, np.ones(N), rcond=None)
    recon = G @ a
    err = np.max(np.abs(recon - 1.0))
    if err < 1e-6:
        # Try to find half-integer approximation
        a_half = np.round(2 * a) / 2
        recon_half = G @ a_half
        if np.max(np.abs(recon_half - 1.0)) < atol:
            return a_half, np.ones(cur_rank), np.ones(cur_rank)
        a_int = np.round(a)
        recon_int = G @ a_int
        if np.max(np.abs(recon_int - 1.0)) < atol:
            return a_int, np.ones(cur_rank), np.ones(cur_rank)
    
    # Try scaling V only
    a_v, res, rank_sub, sv = np.linalg.lstsq(G, np.ones(N), rcond=None)
    recon_v = G @ a_v
    if np.max(np.abs(recon_v - 1.0)) < 1e-6:
        a_v_half = np.round(2 * a_v) / 2
        recon_v_half = G @ a_v_half
        if np.max(np.abs(recon_v_half - 1.0)) < atol:
            return np.ones(cur_rank), a_v_half, np.ones(cur_rank)
    
    # Try scaling W only
    a_w, res, rank_sub, sv = np.linalg.lstsq(G, np.ones(N), rcond=None)
    recon_w = G @ a_w
    if np.max(np.abs(recon_w - 1.0)) < 1e-6:
        a_w_half = np.round(2 * a_w) / 2
        recon_w_half = G @ a_w_half
        if np.max(np.abs(recon_w_half - 1.0)) < atol:
            return np.ones(cur_rank), np.ones(cur_rank), a_w_half
    
    # Try distributed scaling: scale some columns in U, some in V, some in W
    # For each column, try assigning the scaling to U, V, or W
    # This is a combinatorial problem, so use greedy approach
    best_err = np.inf
    best_scalings = None
    
    # Greedy: for each column, pick the factor that gives best fit
    a_U = np.zeros(cur_rank)
    a_V = np.zeros(cur_rank)
    a_W = np.zeros(cur_rank)
    
    for r in range(cur_rank):
        # Try scaling in U
        a_U[r] = 1.0
        recon_u = G @ a_U
        err_u = np.max(np.abs(recon_u - 1.0))
        
        # Try scaling in V
        a_U[r] = 0.0
        a_V[r] = 1.0
        recon_v = G @ a_V
        err_v = np.max(np.abs(recon_v - 1.0))
        
        # Try scaling in W
        a_V[r] = 0.0
        a_W[r] = 1.0
        recon_w = G @ a_W
        err_w = np.max(np.abs(recon_w - 1.0))
        
        # Pick the best
        if err_u <= err_v and err_u <= err_w:
            a_V[r] = 0.0
            a_W[r] = 0.0
        elif err_v <= err_u and err_v <= err_w:
            a_U[r] = 0.0
            a_W[r] = 0.0
        else:
            a_U[r] = 0.0
            a_V[r] = 0.0
    
    # Now solve for the actual scaling values
    # This is still complex, so just return None for now
    return None


def _try_reduce_rank(U, V, W, n, m, p, target_rank, rng, deadline, max_iters=400):
    """
    Attempt to reduce the rank of a valid decomposition down to target_rank.
    Uses rank-revealing QR pivoting to find and eliminate linearly dependent
    columns, with iterative local refinement to preserve the tensor identity.
    Returns (U', V', W', rank') or None on failure.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    cur_rank = U.shape[1]

    if cur_rank <= target_rank:
        return U, V, W, cur_rank

    # Precompute row indices for tensor evaluation
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = (i_idx * m + j_idx).astype(np.int64)
    v_rows = (j_idx * p + k_idx).astype(np.int64)
    w_rows = (k_idx * n + i_idx).astype(np.int64)
    N = len(u_rows)

    # Build the contribution matrix G: shape (N, cur_rank)
    G = np.empty((N, cur_rank), dtype=np.float64)
    for r in range(cur_rank):
        G[:, r] = U[u_rows, r] * V[v_rows, r] * W[w_rows, r]

    # Strategy 1: QR with column pivoting
    Q, R, piv = np.linalg.qr(G, mode='complete', pivoting=True)
    best_cols = piv[:target_rank]

    G_sub = G[:, best_cols]
    a, residuals, rank_sub, sv = np.linalg.lstsq(G_sub, np.ones(N), rcond=None)

    recon = G_sub @ a
    err = np.max(np.abs(recon - 1.0))
    if err < 1e-6:
        # Try to find half-integer scaling
        a_half = np.round(2 * a) / 2
        recon_half = G_sub @ a_half
        if np.max(np.abs(recon_half - 1.0)) < 1e-9:
            U_new = U[:, best_cols] * a_half[None, :]
            V_new = V[:, best_cols].copy()
            W_new = W[:, best_cols].copy()
            if verify_decomposition(U_new, V_new, W_new, n, m, p, atol=1e-9):
                return U_new, V_new, W_new, target_rank
        
        a_int = np.round(a)
        recon_int = G_sub @ a_int
        if np.max(np.abs(recon_int - 1.0)) < 1e-9:
            U_new = U[:, best_cols] * a_int[None, :]
            V_new = V[:, best_cols].copy()
            W_new = W[:, best_cols].copy()
            if verify_decomposition(U_new, V_new, W_new, n, m, p, atol=1e-9):
                return U_new, V_new, W_new, target_rank

    # Strategy 2: Try removing columns one at a time
    for cols_to_remove in range(1, cur_rank - target_rank + 1):
        if time.time() > deadline:
            break
        
        # Try all combinations of columns to remove (only for small numbers)
        if cols_to_remove <= 3:
            from itertools import combinations
            for remove_set in combinations(range(cur_rank), cols_to_remove):
                keep_set = [c for c in range(cur_rank) if c not in remove_set]
                if len(keep_set) != target_rank:
                    continue
                
                G_keep = G[:, keep_set]
                a_keep, res, rank_keep, sv_keep = np.linalg.lstsq(G_keep, np.ones(N), rcond=None)
                recon_keep = G_keep @ a_keep
                if np.max(np.abs(recon_keep - 1.0)) < 1e-6:
                    a_keep_half = np.round(2 * a_keep) / 2
                    recon_keep_half = G_keep @ a_keep_half
                    if np.max(np.abs(recon_keep_half - 1.0)) < 1e-9:
                        U_new = U[:, keep_set] * a_keep_half[None, :]
                        V_new = V[:, keep_set].copy()
                        W_new = W[:, keep_set].copy()
                        if verify_decomposition(U_new, V_new, W_new, n, m, p, atol=1e-9):
                            return U_new, V_new, W_new, target_rank
    
    # Strategy 3: Random subset search
    for _ in range(50):
        if time.time() > deadline:
            break
        
        random_cols = rng.choice(cur_rank, size=target_rank, replace=False)
        G_rand = G[:, random_cols]
        a_rand, res, rank_rand, sv_rand = np.linalg.lstsq(G_rand, np.ones(N), rcond=None)
        recon_rand = G_rand @ a_rand
        if np.max(np.abs(recon_rand - 1.0)) < 1e-6:
            a_rand_half = np.round(2 * a_rand) / 2
            recon_rand_half = G_rand @ a_rand_half
            if np.max(np.abs(recon_rand_half - 1.0)) < 1e-9:
                U_new = U[:, random_cols] * a_rand_half[None, :]
                V_new = V[:, random_cols].copy()
                W_new = W[:, random_cols].copy()
                if verify_decomposition(U_new, V_new, W_new, n, m, p, atol=1e-9):
                    return U_new, V_new, W_new, target_rank

    return None


def _randomized_rank_search(U, V, W, n, m, p, deadline, max_attempts=30):
    """
    Randomized search for rank reduction below the current rank.
    Tries multiple random perturbations and QR-based column elimination.
    Returns best (U, V, W, rank) found, or None.
    """
    cur_rank = U.shape[1]
    best = (U.copy(), V.copy(), W.copy(), cur_rank)
    rng = np.random.default_rng(42)

    # Precompute row indices
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = (i_idx * m + j_idx).astype(np.int64)
    v_rows = (j_idx * p + k_idx).astype(np.int64)
    w_rows = (k_idx * n + i_idx).astype(np.int64)
    N = len(u_rows)

    # Build contribution matrix
    G = np.empty((N, cur_rank), dtype=np.float64)
    for r in range(cur_rank):
        G[:, r] = U[u_rows, r] * V[v_rows, r] * W[w_rows, r]

    # First, try systematic reduction from current rank downward
    for target in range(cur_rank - 1, max(20, cur_rank - 15), -1):
        if time.time() > deadline:
            break

        # Try QR-based approach
        Q, R, piv = np.linalg.qr(G, mode='complete', pivoting=True)
        best_cols = piv[:target]
        G_sub = G[:, best_cols]
        a, res, rank_sub, sv = np.linalg.lstsq(G_sub, np.ones(N), rcond=None)
        recon = G_sub @ a
        if np.max(np.abs(recon - 1.0)) < 1e-6:
            a_half = np.round(2 * a) / 2
            recon_half = G_sub @ a_half
            if np.max(np.abs(recon_half - 1.0)) < 1e-9:
                U_new = U[:, best_cols] * a_half[None, :]
                V_new = V[:, best_cols].copy()
                W_new = W[:, best_cols].copy()
                if verify_decomposition(U_new, V_new, W_new, n, m, p, atol=1e-9):
                    best = (U_new, V_new, W_new, target)
                    cur_rank = target
                    U, V, W = U_new, V_new, W_new
                    # Update G
                    G = np.empty((N, cur_rank), dtype=np.float64)
                    for r in range(cur_rank):
                        G[:, r] = U[u_rows, r] * V[v_rows, r] * W[w_rows, r]
                    break

    # Then try random search for further reduction
    for attempt in range(20):
        if time.time() > deadline:
            break

        for target in range(cur_rank - 1, max(15, cur_rank - 8), -1):
            if time.time() > deadline:
                break

            # Random column selection
            random_cols = rng.choice(cur_rank, size=target, replace=False)
            G_rand = G[:, random_cols]
            a_rand, res, rank_rand, sv_rand = np.linalg.lstsq(G_rand, np.ones(N), rcond=None)
            recon_rand = G_rand @ a_rand
            if np.max(np.abs(recon_rand - 1.0)) < 1e-6:
                a_rand_half = np.round(2 * a_rand) / 2
                recon_rand_half = G_rand @ a_rand_half
                if np.max(np.abs(recon_rand_half - 1.0)) < 1e-9:
                    U_new = U[:, random_cols] * a_rand_half[None, :]
                    V_new = V[:, random_cols].copy()
                    W_new = W[:, random_cols].copy()
                    if verify_decomposition(U_new, V_new, W_new, n, m, p, atol=1e-9):
                        best = (U_new, V_new, W_new, target)
                        cur_rank = target
                        U, V, W = U_new, V_new, W_new
                        G = np.empty((N, cur_rank), dtype=np.float64)
                        for r in range(cur_rank):
                            G[:, r] = U[u_rows, r] * V[v_rows, r] * W[w_rows, r]
                        break

    return best if best[3] < U.shape[1] else None


def schoolbook_lists(n, m, p):
    """Return schoolbook decomposition as nested lists (fallback)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    Uses block-Strassen base (rank 84) + randomized rank-reduction search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s wall-clock

    n, m, p = DIMENSIONS  # 3, 5, 6

    # Step 1: Build block-Strassen decomposition (rank 84)
    n_sizes = [2, 1]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 2]

    assert sum(n_sizes) == n
    assert sum(m_sizes) == m
    assert sum(p_sizes) == p

    U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)

    # Verify correctness
    if not verify_decomposition(U, V, W, n, m, p, atol=1e-9):
        # Should not happen, but fall back to schoolbook
        U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Step 2: Attempt systematic rank reduction
    best_result = None
    try:
        if time.time() < deadline:
            best_result = _randomized_rank_search(U, V, W, n, m, p, deadline, max_attempts=20)
    except Exception:
        best_result = None

    # Step 2b: Try different block partitions if we have time
    if best_result is None and time.time() < deadline - 30:
        # Try alternative block partitions
        alternative_partitions = [
            ([2, 1], [2, 2, 1], [2, 2, 2]),  # current
            ([1, 2], [2, 2, 1], [2, 2, 2]),
            ([2, 1], [1, 2, 2], [2, 2, 2]),
            ([2, 1], [2, 1, 2], [2, 2, 2]),
            ([2, 1], [2, 2, 1], [1, 2, 3]),
            ([2, 1], [2, 2, 1], [2, 1, 3]),
            ([2, 1], [2, 2, 1], [3, 1, 2]),
        ]
        
        for part in alternative_partitions:
            if time.time() > deadline - 10:
                break
            n_sizes, m_sizes, p_sizes = part
            if sum(n_sizes) != n or sum(m_sizes) != m or sum(p_sizes) != p:
                continue
            
            U_alt, V_alt, W_alt, rank_alt = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
            if verify_decomposition(U_alt, V_alt, W_alt, n, m, p, atol=1e-9):
                if best_result is None or rank_alt < best_result[3]:
                    best_result = (U_alt, V_alt, W_alt, rank_alt)

    # Step 3: Return the best valid result
    if best_result is not None:
        U_best, V_best, W_best, rank_best = best_result
        if verify_decomposition(U_best, V_best, W_best, n, m, p, atol=1e-9):
            return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}

    # Fallback: return the block-Strassen decomposition (rank 84)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
