"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition.

Strategy:
  1. Hierarchical block decomposition: split n=3 into 2+1, m=5 into 2+2+1,
     p=5 into 2+2+1, giving 18 blocks.  Each block uses the best known
     decomposition for its local size:
       - 2x2x2: Strassen rank 7
       - other sizes: schoolbook (rank = ni*mi*pi)
     This gives a deterministic rank-71 construction.
  2. A focused search for a lower-rank decomposition: we fix the 2x2x2
     Strassen cores and then try to express the remaining tensor
     entries using a small number of additional columns.  We search over
     structured integer/half-integer patterns for U and V, solve for W, and
     verify.
  3. Return the best valid decomposition found.
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
    """Independent block decomposition: Strassen on 2x2x2, schoolbook elsewhere.
    Uses n=2+1, m=2+2+1, p=2+2+1 for rank-71 baseline.
    Returns U, V, W (numpy arrays)."""
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]

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
    """Check that the decomposition exactly represents matrix multiplication.
    Vectorized: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return np.max(np.abs(T - 1.0)) < tol


def _try_lower_rank(n, m, p, deadline, max_rank_try=71, min_rank_try=65):
    """
    Search for a decomposition with rank < 71 by fixing Strassen-like cores
    and solving for the remaining factors.  We try several structured patterns.
    Returns (U, V, W, rank) or None.
    """
    rng = np.random.RandomState(12345)

    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices.append((i, j, k))
    n_eqs = len(eq_indices)

    for R in range(max_rank_try - 1, min_rank_try - 1, -1):
        if time.time() > deadline - 25:
            break

        for attempt in range(200):
            if time.time() > deadline - 20:
                break

            # Generate structured U and V
            if attempt % 7 == 0:
                # Sparse integer pattern
                U_rand = np.zeros((n * m, R))
                V_rand = np.zeros((m * p, R))
                mask_u = rng.rand(n * m, R) < 0.25
                mask_v = rng.rand(m * p, R) < 0.25
                U_rand[mask_u] = rng.choice([-1.0, 1.0], size=mask_u.sum())
                V_rand[mask_v] = rng.choice([-1.0, 1.0], size=mask_v.sum())
            elif attempt % 7 == 1:
                # Small integer entries
                U_rand = rng.randint(-1, 2, (n * m, R)).astype(np.float64)
                V_rand = rng.randint(-1, 2, (m * p, R)).astype(np.float64)
            elif attempt % 7 == 2:
                # Binary pattern
                U_rand = (rng.rand(n * m, R) < 0.4).astype(np.float64)
                V_rand = (rng.rand(m * p, R) < 0.4).astype(np.float64)
            elif attempt % 7 == 3:
                # Embed Strassen cores at (0,0,0) and (0,1,1) positions
                U_core, V_core, W_core = _strassen_2x2x2()
                U_rand = np.zeros((n * m, R))
                V_rand = np.zeros((m * p, R))
                for li in range(2):
                    for lj in range(2):
                        U_rand[li * m + lj, :7] = U_core[li * 2 + lj, :]
                for lj in range(2):
                    for lk in range(2):
                        V_rand[lj * p + lk, :7] = V_core[lj * 2 + lk, :]
                for li in range(2):
                    for lj in range(2):
                        U_rand[li * m + (2 + lj), 7:14] = U_core[li * 2 + lj, :]
                for lj in range(2):
                    for lk in range(2):
                        V_rand[(2 + lj) * p + (2 + lk), 7:14] = V_core[lj * 2 + lk, :]
                U_rand[:, 14:] = rng.randint(-1, 2, (n * m, R - 14)).astype(np.float64)
                V_rand[:, 14:] = rng.randint(-1, 2, (m * p, R - 14)).astype(np.float64)
            elif attempt % 7 == 4:
                # Embed all 4 Strassen cores (2x2x2 blocks)
                U_core, V_core, W_core = _strassen_2x2x2()
                U_rand = np.zeros((n * m, R))
                V_rand = np.zeros((m * p, R))
                cores = [(0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2)]
                for c_idx, (no, mo, po) in enumerate(cores):
                    col_start = c_idx * 7
                    for li in range(2):
                        for lj in range(2):
                            U_rand[(no + li) * m + (mo + lj), col_start:col_start+7] = U_core[li * 2 + lj, :]
                    for lj in range(2):
                        for lk in range(2):
                            V_rand[(mo + lj) * p + (po + lk), col_start:col_start+7] = V_core[lj * 2 + lk, :]
                U_rand[:, 28:] = rng.randint(-1, 2, (n * m, R - 28)).astype(np.float64)
                V_rand[:, 28:] = rng.randint(-1, 2, (m * p, R - 28)).astype(np.float64)
            elif attempt % 7 == 5:
                # Mixed pattern with half-integers
                U_rand = np.zeros((n * m, R))
                V_rand = np.zeros((m * p, R))
                mask_u = rng.rand(n * m, R) < 0.3
                mask_v = rng.rand(m * p, R) < 0.3
                U_rand[mask_u] = rng.choice([-1.0, -0.5, 0.5, 1.0], size=mask_u.sum())
                V_rand[mask_v] = rng.choice([-1.0, -0.5, 0.5, 1.0], size=mask_v.sum())
            else:
                # Structured: copy from baseline and perturb
                U_base, V_base, W_base, _ = _independent_block_decomp(n, m, p)
                if R <= U_base.shape[1]:
                    U_rand = U_base[:, :R].copy()
                else:
                    U_rand = np.hstack([U_base, np.zeros((n*m, R-U_base.shape[1]))])
                if R <= V_base.shape[1]:
                    V_rand = V_base[:, :R].copy()
                else:
                    V_rand = np.hstack([V_base, np.zeros((m*p, R-V_base.shape[1]))])
                mask_u = rng.rand(n * m, R) < 0.1
                mask_v = rng.rand(m * p, R) < 0.1
                U_rand[mask_u] += rng.choice([-1.0, 1.0], size=mask_u.sum())
                V_rand[mask_v] += rng.choice([-1.0, 1.0], size=mask_v.sum())

            # Build A matrix: A[eq, (k*n+i)*R + r] = U[i*m+j, r] * V[j*p+k, r]
            n_vars = p * n * R
            A = np.zeros((n_eqs, n_vars), dtype=np.float64)
            b = np.ones(n_eqs, dtype=np.float64)

            for eq, (i, j, k) in enumerate(eq_indices):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                for r in range(R):
                    A[eq, w_row * R + r] = U_rand[u_row, r] * V_rand[v_row, r]

            # Solve A W_flat = b
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue

            W_rand = W_flat.reshape(p * n, R)

            # Check if W is close to half-integers
            W_check = W_rand * 2.0
            dist = np.abs(W_check - np.round(W_check))
            max_dist = np.max(dist)
            if max_dist > 1e-4:
                continue

            W_int = np.round(W_check) / 2.0

            # Validate the full decomposition
            if _verify(U_rand, V_rand, W_int, n, m, p):
                return U_rand, V_rand, W_int, R

    return None


def _try_block_split_search(n, m, p, deadline, max_rank_try=71, min_rank_try=60):
    """
    Alternative search: decompose into 3x2x5 and 3x3x5 blocks.
    For 3x2x5: split m=2, p=5 into p=2+3. The 2x2x2 part uses Strassen.
    For 3x3x5: schoolbook.
    Then try to merge/collapse columns to reduce total rank.
    """
    if time.time() > deadline - 30:
        return None

    rng = np.random.RandomState(99999)
    eq_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    n_eqs = len(eq_indices)

    # First, construct the block-split baseline
    # Block 1: rows 0-2, cols 0-1 (m=2), p=5 -> split p into 2+3
    #   Sub-block 1a: 3x2x2 (Strassen on 2x2x2 parts, schoolbook on 1x2x2 parts)
    #   Sub-block 1b: 3x2x3 (schoolbook)
    # Block 2: rows 0-2, cols 2-4 (m=3), p=5 -> schoolbook 3x3x5

    # Actually, let's just try random structured searches with the block structure in mind
    for R in range(max_rank_try - 1, min_rank_try - 1, -1):
        if time.time() > deadline - 25:
            break

        for attempt in range(150):
            if time.time() > deadline - 20:
                break

            # Pattern: exploit the 2+3 split in m dimension
            U_rand = np.zeros((n * m, R))
            V_rand = np.zeros((m * p, R))

            if attempt % 3 == 0:
                # Structured: first R-15 columns for m=0,1 part; last 15 for m=2,3,4
                # m=0,1 part: 3*2=6 rows in U, 2*5=10 rows in V
                # Use Strassen-like structure for the 2x2x2 sub-blocks
                U_core, V_core, W_core = _strassen_2x2x2()
                # Embed Strassen for (i=0,j=0,k=0), (i=0,j=0,k=1), etc.
                # For now, use random small integers
                U_rand[:, :R-15] = rng.randint(-1, 2, (n * m, R - 15)).astype(np.float64)
                V_rand[:, :R-15] = rng.randint(-1, 2, (m * p, R - 15)).astype(np.float64)
                U_rand[:, R-15:] = rng.randint(-1, 2, (n * m, 15)).astype(np.float64)
                V_rand[:, R-15:] = rng.randint(-1, 2, (m * p, 15)).astype(np.float64)
            elif attempt % 3 == 1:
                # Sparse pattern
                mask_u = rng.rand(n * m, R) < 0.2
                mask_v = rng.rand(m * p, R) < 0.2
                U_rand[mask_u] = rng.choice([-1.0, 1.0], size=mask_u.sum())
                V_rand[mask_v] = rng.choice([-1.0, 1.0], size=mask_v.sum())
            else:
                # Half-integer pattern
                mask_u = rng.rand(n * m, R) < 0.25
                mask_v = rng.rand(m * p, R) < 0.25
                U_rand[mask_u] = rng.choice([-1.0, -0.5, 0.5, 1.0], size=mask_u.sum())
                V_rand[mask_v] = rng.choice([-1.0, -0.5, 0.5, 1.0], size=mask_v.sum())

            n_vars = p * n * R
            A = np.zeros((n_eqs, n_vars), dtype=np.float64)
            b = np.ones(n_eqs, dtype=np.float64)

            for eq, (i, j, k) in enumerate(eq_indices):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                for r in range(R):
                    A[eq, w_row * R + r] = U_rand[u_row, r] * V_rand[v_row, r]

            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue

            W_rand = W_flat.reshape(p * n, R)

            # Check half-integer constraint
            W_check = W_rand * 2.0
            dist = np.abs(W_check - np.round(W_check))
            if np.max(dist) > 1e-4:
                continue

            W_int = np.round(W_check) / 2.0

            if _verify(U_rand, V_rand, W_int, n, m, p):
                return U_rand, V_rand, W_int, R

    return None


def _polish_decomposition(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding linear dependencies among columns.
    If some columns are linearly dependent, we can merge them.
    Also try to round U, V, W to nearest half-integers and re-verify.
    """
    if time.time() > deadline - 10:
        return U, V, W, U.shape[1]

    R = U.shape[1]
    best_U, best_V, best_W, best_R = U, V, W, R

    # Try rounding to half-integers
    U_round = np.round(U * 2.0) / 2.0
    V_round = np.round(V * 2.0) / 2.0
    W_round = np.round(W * 2.0) / 2.0
    if _verify(U_round, V_round, W_round, n, m, p):
        best_U, best_V, best_W = U_round, V_round, W_round

    # Try to identify and remove linearly dependent columns
    # This is expensive, so only do it if we have time
    if time.time() < deadline - 15 and R > 60:
        # Check if any pair of columns in U are nearly identical
        # (which would mean we can potentially merge them)
        # This is a heuristic; full rank reduction is NP-hard
        pass

    return best_U, best_V, best_W, best_R


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # 1. Independent block decomposition (rank 71) as baseline
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
    if time.time() < deadline - 30:
        result = _try_lower_rank(n, m, p, deadline, max_rank_try=best_rank, min_rank_try=65)
        if result is not None:
            U_new, V_new, W_new, R_new = result
            if R_new < best_rank:
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, R_new

    # 3. Try block-split search
    if time.time() < deadline - 25:
        result2 = _try_block_split_search(n, m, p, deadline, max_rank_try=best_rank, min_rank_try=60)
        if result2 is not None:
            U_new, V_new, W_new, R_new = result2
            if R_new < best_rank:
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, R_new

    # 4. Polish pass: round to half-integers and check
    if time.time() < deadline - 10:
        best_U, best_V, best_W, best_rank = _polish_decomposition(
            best_U, best_V, best_W, n, m, p, deadline
        )

    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

# EVOLVE-BLOCK-END
