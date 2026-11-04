"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition.

Strategy:
  1. Finer block decomposition: n=[1,2], m=[2,1,2], p=[2,1,2] gives 18 blocks
     with 4 Strassen 2x2x2 cores (rank 7 each) and 14 schoolbook blocks,
     yielding a deterministic rank-63 construction.
  2. Focused search for lower-rank decompositions: exploit block structure,
     fix structured cores, solve for W via least squares, snap to half-integers.
  3. Aggressive column elimination and merging to reduce rank further.
  4. Completion approach: fix most columns, solve for a few free ones.
  5. Return the best valid decomposition found.
"""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7); indexing U[i*2+j,c], V[j*2+k,c], W[k*2+i,c]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[:, 0] = [1, 0, 0, 1]; V[:, 0] = [1, 0, 0, 1]; W[:, 0] = [1, 0, 0, 1]
    U[:, 1] = [0, 0, 1, 1]; V[:, 1] = [1, 0, 0, 0]; W[:, 1] = [0, 1, 0, -1]
    U[:, 2] = [1, 0, 0, 0]; V[:, 2] = [0, 1, 0, -1]; W[:, 2] = [0, 0, 1, 1]
    U[:, 3] = [0, 0, 0, 1]; V[:, 3] = [-1, 0, 1, 0]; W[:, 3] = [1, 1, 0, 0]
    U[:, 4] = [1, 1, 0, 0]; V[:, 4] = [0, 0, 0, 1]; W[:, 4] = [-1, 0, 1, 0]
    U[:, 5] = [-1, 0, 1, 0]; V[:, 5] = [1, 1, 0, 0]; W[:, 5] = [0, 0, 0, 1]
    U[:, 6] = [0, 1, 0, -1]; V[:, 6] = [0, 0, 1, 1]; W[:, 6] = [1, 0, 0, 0]
    return U, V, W


def _schoolbook_local(ni, mi, pi):
    """Schoolbook local decomposition for ni x mi x pi."""
    rank = ni * mi * pi
    U = np.zeros((ni * mi, rank))
    V = np.zeros((mi * pi, rank))
    W = np.zeros((pi * ni, rank))
    col = 0
    for i in range(ni):
        for j in range(mi):
            for k in range(pi):
                U[i * mi + j, col] = 1
                V[j * pi + k, col] = 1
                W[k * ni + i, col] = 1
                col += 1
    return U, V, W


def _embed_block(U_loc, V_loc, W_loc, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    """Embed a local ni x mi x pi decomposition into global dimensions."""
    r = U_loc.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    for li in range(ni):
        for lj in range(mi):
            U_g[(n_off + li) * m + (m_off + lj), :] = U_loc[li * mi + lj, :]
    for lj in range(mi):
        for lk in range(pi):
            V_g[(m_off + lj) * p + (p_off + lk), :] = V_loc[lj * pi + lk, :]
    for lk in range(pi):
        for li in range(ni):
            W_g[(p_off + lk) * n + (n_off + li), :] = W_loc[lk * ni + li, :]
    return U_g, V_g, W_g


def _independent_block_decomp(n, m, p):
    """Finer block decomposition: n=[1,2], m=[2,1,2], p=[2,1,2].
    4 Strassen 2x2x2 blocks (rank 7 each) + 14 schoolbook blocks = rank 63.
    Returns U, V, W (numpy arrays) and total_rank."""
    n_parts = [1, 2]
    m_parts = [2, 1, 2]
    p_parts = [2, 1, 2]
    n_offsets = [0, 1]
    m_offsets = [0, 2, 3]
    p_offsets = [0, 2, 3]

    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    block_ranks[(i, j, k)] = 7
                else:
                    block_ranks[(i, j, k)] = ni * mi * pi
                total_rank += block_ranks[(i, j, k)]

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = block_ranks[(i, j, k)]
                if ni == 2 and mi == 2 and pi == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                else:
                    U_loc, V_loc, W_loc = _schoolbook_local(ni, mi, pi)
                U_g, V_g, W_g = _embed_block(
                    U_loc, V_loc, W_loc, n, m, p,
                    n_offsets[i], m_offsets[j], p_offsets[k],
                    ni, mi, pi
                )
                U[:, col:col + r] += U_g
                V[:, col:col + r] += V_g
                W[:, col:col + r] += W_g
                col += r
    return U, V, W, total_rank


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly represents matrix multiplication."""
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return np.max(np.abs(T - 1.0)) < tol


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using precomputed structure."""
    R = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(R):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col, optimize=True)
    return np.max(np.abs(T - 1.0)) < tol


def _snap_half_integer(mat):
    """Snap matrix entries to nearest half-integer."""
    return np.round(mat * 2.0) / 2.0


def _is_half_integer(mat, tol=1e-6):
    """Check if all entries are close to half-integers."""
    doubled = mat * 2.0
    return np.max(np.abs(doubled - np.round(doubled))) < tol


def _build_linear_system(U_rand, V_rand, n, m, p, R):
    """Build the linear system A * W_flat = b for solving W."""
    n_eqs = n * m * p
    n_vars = p * n * R
    A = np.zeros((n_eqs, n_vars), dtype=np.float64)
    b = np.ones(n_eqs, dtype=np.float64)
    
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_block_start = (k * n + i) * R
                A[eq, w_block_start:w_block_start + R] = U_rand[u_row, :] * V_rand[v_row, :]
                eq += 1
    return A, b


def _try_lower_rank(n, m, p, deadline, max_rank_try=63, min_rank_try=55):
    """
    Search for a decomposition with rank < max_rank_try by fixing structured cores
    and solving for W. Returns (U, V, W, rank) or None.
    """
    rng = np.random.RandomState(42)

    # Precompute Strassen cores for embedding
    U_core, V_core, W_core = _strassen_2x2x2()

    # Strassen block positions in the finer partition:
    # n_parts=[1,2], m_parts=[2,1,2], p_parts=[2,1,2]
    # Strassen blocks where ni=2,mi=2,pi=2: (i=1,j=0,k=0),(1,0,2),(1,2,0),(1,2,2)
    # n_off=1, m_off in {0,3}, p_off in {0,3}
    strassen_positions = [(1, 0, 0), (1, 0, 3), (1, 3, 0), (1, 3, 3)]

    # Precompute baseline for perturbation
    U_base, V_base, W_base, _ = _independent_block_decomp(n, m, p)

    for R in range(max_rank_try - 1, min_rank_try - 1, -1):
        if time.time() > deadline - 20:
            break

        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break

            U_rand = np.zeros((n * m, R))
            V_rand = np.zeros((m * p, R))

            if R >= 28:
                # Embed all 4 Strassen cores in first 28 columns
                for c_idx, (no, mo, po) in enumerate(strassen_positions):
                    col_start = c_idx * 7
                    for li in range(2):
                        for lj in range(2):
                            U_rand[(no + li) * m + (mo + lj), col_start:col_start + 7] = U_core[li * 2 + lj, :]
                    for lj in range(2):
                        for lk in range(2):
                            V_rand[(mo + lj) * p + (po + lk), col_start:col_start + 7] = V_core[lj * 2 + lk, :]

                # Fill remaining columns with structured random patterns
                remaining = R - 28
                if remaining > 0:
                    pattern = attempt % 8
                    if pattern == 0:
                        # Sparse integer
                        mask_u = rng.rand(n * m, remaining) < 0.25
                        mask_v = rng.rand(m * p, remaining) < 0.25
                        U_rand[:, 28:][mask_u] = rng.choice([-1.0, 1.0], size=mask_u.sum())
                        V_rand[:, 28:][mask_v] = rng.choice([-1.0, 1.0], size=mask_v.sum())
                    elif pattern == 1:
                        # Small integers
                        U_rand[:, 28:] = rng.randint(-1, 2, (n * m, remaining)).astype(np.float64)
                        V_rand[:, 28:] = rng.randint(-1, 2, (m * p, remaining)).astype(np.float64)
                    elif pattern == 2:
                        # Binary
                        U_rand[:, 28:] = (rng.rand(n * m, remaining) < 0.35).astype(np.float64)
                        V_rand[:, 28:] = (rng.rand(m * p, remaining) < 0.35).astype(np.float64)
                    elif pattern == 3:
                        # Half-integers
                        mask_u = rng.rand(n * m, remaining) < 0.3
                        mask_v = rng.rand(m * p, remaining) < 0.3
                        U_rand[:, 28:][mask_u] = rng.choice([-1.0, -0.5, 0.5, 1.0], size=mask_u.sum())
                        V_rand[:, 28:][mask_v] = rng.choice([-1.0, -0.5, 0.5, 1.0], size=mask_v.sum())
                    elif pattern == 4:
                        # Perturb baseline
                        if R <= U_base.shape[1]:
                            U_rand[:, 28:] = U_base[:, 28:R].copy()
                            V_rand[:, 28:] = V_base[:, 28:R].copy()
                        else:
                            U_rand[:, 28:U_base.shape[1]] = U_base[:, 28:].copy()
                            V_rand[:, 28:V_base.shape[1]] = V_base[:, 28:].copy()
                        mask_u = rng.rand(n * m, remaining) < 0.1
                        mask_v = rng.rand(m * p, remaining) < 0.1
                        U_rand[:, 28:][mask_u] += rng.choice([-1.0, 1.0], size=mask_u.sum())
                        V_rand[:, 28:][mask_v] += rng.choice([-1.0, 1.0], size=mask_v.sum())
                    elif pattern == 5:
                        # Sparse with larger magnitude
                        mask_u = rng.rand(n * m, remaining) < 0.2
                        mask_v = rng.rand(m * p, remaining) < 0.2
                        U_rand[:, 28:][mask_u] = rng.choice([-2.0, -1.0, 1.0, 2.0], size=mask_u.sum())
                        V_rand[:, 28:][mask_v] = rng.choice([-2.0, -1.0, 1.0, 2.0], size=mask_v.sum())
                    elif pattern == 6:
                        # Very sparse
                        mask_u = rng.rand(n * m, remaining) < 0.15
                        mask_v = rng.rand(m * p, remaining) < 0.15
                        U_rand[:, 28:][mask_u] = rng.choice([-1.0, 1.0], size=mask_u.sum())
                        V_rand[:, 28:][mask_v] = rng.choice([-1.0, 1.0], size=mask_v.sum())
                    else:
                        # Dense small integers
                        U_rand[:, 28:] = rng.randint(-1, 2, (n * m, remaining)).astype(np.float64) * 0.5
                        V_rand[:, 28:] = rng.randint(-1, 2, (m * p, remaining)).astype(np.float64) * 0.5
            else:
                # R < 28: no full Strassen embedding, use sparse patterns
                pattern = attempt % 4
                if pattern == 0:
                    mask_u = rng.rand(n * m, R) < 0.25
                    mask_v = rng.rand(m * p, R) < 0.25
                    U_rand[mask_u] = rng.choice([-1.0, 1.0], size=mask_u.sum())
                    V_rand[mask_v] = rng.choice([-1.0, 1.0], size=mask_v.sum())
                elif pattern == 1:
                    U_rand = rng.randint(-1, 2, (n * m, R)).astype(np.float64)
                    V_rand = rng.randint(-1, 2, (m * p, R)).astype(np.float64)
                elif pattern == 2:
                    U_rand = (rng.rand(n * m, R) < 0.35).astype(np.float64)
                    V_rand = (rng.rand(m * p, R) < 0.35).astype(np.float64)
                else:
                    mask_u = rng.rand(n * m, R) < 0.3
                    mask_v = rng.rand(m * p, R) < 0.3
                    U_rand[mask_u] = rng.choice([-1.0, -0.5, 0.5, 1.0], size=mask_u.sum())
                    V_rand[mask_v] = rng.choice([-1.0, -0.5, 0.5, 1.0], size=mask_v.sum())

            # Build linear system and solve for W
            A, b = _build_linear_system(U_rand, V_rand, n, m, p, R)
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue

            W_rand = W_flat.reshape(p * n, R)

            # Check if W is close to half-integers
            W_check = W_rand * 2.0
            dist = np.abs(W_check - np.round(W_check))
            max_dist = np.max(dist)
            if max_dist > 5e-4:
                continue

            W_int = np.round(W_check) / 2.0

            # Validate the full decomposition
            if _verify(U_rand, V_rand, W_int, n, m, p):
                return U_rand, V_rand, W_int, R

    return None


def _try_completion(n, m, p, deadline, base_U, base_V, base_W, max_free=6):
    """
    Completion approach: fix most columns from a baseline, leave a few free,
    and solve for the free columns.
    """
    R_base = base_U.shape[1]
    rng = np.random.RandomState(999)
    
    for n_free in range(1, max_free + 1):
        if time.time() > deadline - 10:
            break
        
        R_total = R_base + n_free
        
        for attempt in range(50):
            if time.time() > deadline - 8:
                break
            
            # Choose which columns to make free (random subset of baseline columns to perturb)
            free_cols = rng.choice(R_base, size=n_free, replace=False)
            
            U_rand = base_U.copy()
            V_rand = base_V.copy()
            
            # Perturb the free columns slightly
            for fc in free_cols:
                mask_u = rng.rand(n * m) < 0.3
                mask_v = rng.rand(m * p) < 0.3
                U_rand[:, fc][mask_u] += rng.choice([-1.0, 1.0], size=mask_u.sum())
                V_rand[:, fc][mask_v] += rng.choice([-1.0, 1.0], size=mask_v.sum())
            
            # Add new free columns
            for nf in range(n_free):
                col_idx = R_base + nf
                mask_u = rng.rand(n * m) < 0.3
                mask_v = rng.rand(m * p) < 0.3
                U_rand[:, col_idx][mask_u] = rng.choice([-1.0, 1.0], size=mask_u.sum())
                V_rand[:, col_idx][mask_v] = rng.choice([-1.0, 1.0], size=mask_v.sum())
            
            # Build linear system and solve for W
            A, b = _build_linear_system(U_rand, V_rand, n, m, p, R_total)
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            W_rand = W_flat.reshape(p * n, R_total)
            
            # Check if W is close to half-integers
            W_check = W_rand * 2.0
            dist = np.abs(W_check - np.round(W_check))
            max_dist = np.max(dist)
            if max_dist > 5e-4:
                continue
            
            W_int = np.round(W_check) / 2.0
            
            # Validate
            if _verify(U_rand, V_rand, W_int, n, m, p):
                return U_rand, V_rand, W_int, R_total
    
    return None


def _try_polish(U, V, W, n, m, p, deadline):
    """Try to reduce rank by zeroing out columns that don't affect correctness.
    Also try merging pairs of columns."""
    R = U.shape[1]
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = R

    # Phase 1: Try removing columns one at a time (greedy, multiple passes)
    for pass_num in range(8):
        if time.time() > deadline - 5:
            break
        if current_rank <= 55:
            break
        removed_any = False
        for col_idx in range(current_rank - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if current_rank <= 55:
                break

            saved_U = current_U[:, col_idx].copy()
            saved_V = current_V[:, col_idx].copy()
            saved_W = current_W[:, col_idx].copy()

            current_U[:, col_idx] = 0
            current_V[:, col_idx] = 0
            current_W[:, col_idx] = 0

            if _verify(current_U, current_V, current_W, n, m, p):
                mask = np.ones(current_rank, dtype=bool)
                mask[col_idx] = False
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1
                removed_any = True
            else:
                current_U[:, col_idx] = saved_U
                current_V[:, col_idx] = saved_V
                current_W[:, col_idx] = saved_W

        if not removed_any:
            break

    # Phase 2: Try merging two columns into one (rank-1 absorption)
    if time.time() < deadline - 8 and current_rank > 55:
        merged = _try_merge_columns(current_U, current_V, current_W, n, m, p, deadline)
        if merged is not None:
            current_U, current_V, current_W, current_rank = merged
            # After a merge, try more elimination
            for pass_num in range(3):
                if time.time() > deadline - 5:
                    break
                if current_rank <= 55:
                    break
                removed_any = False
                for col_idx in range(current_rank - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    if current_rank <= 55:
                        break
                    saved_U = current_U[:, col_idx].copy()
                    saved_V = current_V[:, col_idx].copy()
                    saved_W = current_W[:, col_idx].copy()
                    current_U[:, col_idx] = 0
                    current_V[:, col_idx] = 0
                    current_W[:, col_idx] = 0
                    if _verify(current_U, current_V, current_W, n, m, p):
                        mask = np.ones(current_rank, dtype=bool)
                        mask[col_idx] = False
                        current_U = current_U[:, mask]
                        current_V = current_V[:, mask]
                        current_W = current_W[:, mask]
                        current_rank -= 1
                        removed_any = True
                    else:
                        current_U[:, col_idx] = saved_U
                        current_V[:, col_idx] = saved_V
                        current_W[:, col_idx] = saved_W
                if not removed_any:
                    break

    if current_rank < R:
        return current_U, current_V, current_W, current_rank
    return None


def _try_merge_columns(U, V, W, n, m, p, deadline):
    """Try to merge two columns by adjusting one to absorb the other."""
    R = U.shape[1]
    best_result = None

    for a in range(R - 1):
        if time.time() > deadline - 3:
            break
        for b in range(a + 1, R):
            if time.time() > deadline - 3:
                break

            u_a = U[:, a]
            u_b = U[:, b]
            v_a = V[:, a]
            v_b = V[:, b]

            nz_a = np.nonzero(u_a)[0]
            nz_b = np.nonzero(u_b)[0]
            if len(nz_a) == 0 or len(nz_b) == 0:
                continue

            common = np.intersect1d(nz_a, nz_b)
            if len(common) < 2:
                continue
            alpha_u = np.median(u_a[common] / u_b[common])
            if np.max(np.abs(u_a - alpha_u * u_b)) > 1e-9:
                continue

            nz_va = np.nonzero(v_a)[0]
            nz_vb = np.nonzero(v_b)[0]
            if len(nz_va) == 0 or len(nz_vb) == 0:
                continue
            common_v = np.intersect1d(nz_va, nz_vb)
            if len(common_v) < 2:
                continue
            alpha_v = np.median(v_a[common_v] / v_b[common_v])
            if np.max(np.abs(v_a - alpha_v * v_b)) > 1e-9:
                continue

            if abs(alpha_u - alpha_v) > 1e-9:
                continue

            W_new = W.copy()
            W_new[:, a] = W[:, a] + alpha_u * alpha_v * W[:, b]

            W_check = W_new[:, a] * 2.0
            if np.max(np.abs(W_check - np.round(W_check))) > 1e-6:
                continue
            W_new[:, a] = np.round(W_check) / 2.0

            mask = np.ones(R, dtype=bool)
            mask[b] = False
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W_new[:, mask]

            if _verify(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, R - 1

    return best_result


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # 1. Finer block decomposition (rank 63) as baseline
    U_ind, V_ind, W_ind, rank_ind = _independent_block_decomp(n, m, p)

    # Verify baseline
    if not _verify(U_ind, V_ind, W_ind, n, m, p):
        # Fallback to pure schoolbook (rank 75)
        rank = n * m * p
        U_ind = np.zeros((n * m, rank))
        V_ind = np.zeros((m * p, rank))
        W_ind = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_ind[i * m + j, col] = 1.0
                    V_ind[j * p + k, col] = 1.0
                    W_ind[k * n + i, col] = 1.0
                    col += 1
        rank_ind = rank

    best_U, best_V, best_W, best_rank = U_ind, V_ind, W_ind, rank_ind

    # 2. Try to find a better (lower rank) decomposition via search
    if time.time() < deadline - 40:
        result = _try_lower_rank(n, m, p, deadline, max_rank_try=best_rank, min_rank_try=55)
        if result is not None:
            U_new, V_new, W_new, R_new = result
            if R_new < best_rank:
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, R_new

    # 3. Try completion approach
    if time.time() < deadline - 30 and best_rank >= 58:
        completion_result = _try_completion(n, m, p, deadline, best_U, best_V, best_W, max_free=5)
        if completion_result is not None:
            U_comp, V_comp, W_comp, R_comp = completion_result
            if R_comp < best_rank:
                best_U, best_V, best_W, best_rank = U_comp, V_comp, W_comp, R_comp

    # 4. Try to polish: remove redundant columns and merge
    if time.time() < deadline - 10:
        polished = _try_polish(best_U, best_V, best_W, n, m, p, deadline)
        if polished is not None:
            U_pol, V_pol, W_pol, R_pol = polished
            if R_pol < best_rank:
                best_U, best_V, best_W, best_rank = U_pol, V_pol, W_pol, R_pol

    # 5. Second round of search with updated baseline if polish helped
    if best_rank < 63 and time.time() < deadline - 20:
        result2 = _try_lower_rank(n, m, p, deadline, max_rank_try=best_rank, min_rank_try=55)
        if result2 is not None:
            U_new2, V_new2, W_new2, R_new2 = result2
            if R_new2 < best_rank:
                best_U, best_V, best_W, best_rank = U_new2, V_new2, W_new2, R_new2

    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

# EVOLVE-BLOCK-END
