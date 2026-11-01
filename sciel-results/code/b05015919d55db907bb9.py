"""Rewrite: investigate structural rank reduction within the Strassen family for 2x4x8."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_terms():
    """Return the 7 Strassen factors for 2x2x2 multiplication."""
    terms = [
        (
            {(0,0): 1, (1,1): 1},   # M1
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},   # M2
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        (
            {(0,0): 1},             # M3
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(1,1): 1},             # M4
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},   # M5
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},  # M6
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},  # M7
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    return terms

def _construct_strassen_56():
    """
    Standard rank-56 construction: 2 A-blocks x 4 B-blocks x 7 Strassen terms.
    
    For 2x4 * 4x8:
    - Partition A's 4 columns into 2 blocks of 2: a_block in {0,1}
    - Partition B's 8 columns into 4 blocks of 2: b_block in {0,1,2,3}
    - For each (a_block, b_block) pair, apply Strassen to the 2x2 by 2x2 product
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen = _strassen_terms()
    col = 0
    for a_blk in range(2):
        for b_blk in range(4):
            for A_c, B_c, C_c in strassen:
                for (li, lj), c in A_c.items():
                    gi, gj = li, 2 * a_blk + lj
                    U[gi * m + gj, col] += c
                for (lj, lk), c in B_c.items():
                    gj, gk = 2 * a_blk + lj, 2 * b_blk + lk
                    V[gj * p + gk, col] += c
                for (li, lk), c in C_c.items():
                    gi, gk = li, 2 * b_blk + lk
                    W[gk * n + gi, col] += c
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _check_column_dependence(U, V, W, n, m, p):
    """
    Check if any column can be expressed as a linear combination of others.
    A column c is dependent if there exist scalars s[0..rank-1] (not all zero)
    with s[c] = 1 such that for all (i,j,k):
      sum_s s[c'] * U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c'] = 0
    
    This is equivalent to checking if the rank of the set of "outer product vectors"
    is less than the number of columns.
    
    Returns list of indices of dependent columns (empty if none).
    """
    rank = len(U[0])
    # Build the tensor as a set of vectors in R^(n*m*p)
    # For each column c, the vector is the entrywise product U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # But more precisely, for each (i,j,k), the coefficient is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    total_dim = n * m * p
    mat = np.zeros((total_dim, rank), dtype=float)
    
    for idx in range(total_dim):
        i = idx // (m * p)
        rem = idx % (m * p)
        j = rem // p
        k = rem % p
        for c in range(rank):
            mat[idx, c] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
    
    # Find columns that are linear combinations of others
    # Compute rank of mat
    from scipy.linalg import svd
    U_sv, s_vals, Vt = svd(mat, full_matrices=False)
    
    # Rank is number of significant singular values
    tol = 1e-9
    r = np.sum(s_vals > tol)
    
    if r < rank:
        # Some columns are dependent. Find which ones.
        # The null space of mat^T gives the dependencies.
        # mat^T is rank x total_dim, null space dimension = rank - r
        # Actually, we need null space of mat (total_dim x rank)
        # Null space vectors v in R^rank such that mat @ v = 0
        # Use the singular vectors corresponding to zero singular values
        if r < rank:
            null_basis = Vt[r:]  # (rank - r, rank)
            # Each row of null_basis is a dependency
            # A column index c is "removable" if there's a dependency where that column has nonzero coefficient
            # and we can express that column in terms of others
            dependent = []
            for row in null_basis:
                for c in range(rank):
                    if abs(row[c]) > tol and c not in dependent:
                        dependent.append(c)
            return dependent
    return []

def _try_rank_reduction_via_nullspace(U, V, W, n, m, p, max_iter=200):
    """
    Attempt to reduce rank by removing dependent columns.
    Uses iterative linear algebra to find removable columns.
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_rank = len(U_arr[0])
    if current_rank == 0:
        return None
    
    remaining = list(range(current_rank))
    
    for iteration in range(max_iter):
        if len(remaining) <= 1:
            break
        
        # Build the tensor matrix for current remaining columns
        total_dim = n * m * p
        sub_rank = len(remaining)
        mat = np.zeros((total_dim, sub_rank), dtype=float)
        
        for idx in range(total_dim):
            i = idx // (m * p)
            rem = idx % (m * p)
            j = rem // p
            k = rem % p
            for nc, orig_c in enumerate(remaining):
                mat[idx, nc] = (U_arr[i * m + j, orig_c] * 
                               V_arr[j * p + k, orig_c] * 
                               W_arr[k * n + i, orig_c])
        
        from scipy.linalg import svd
        try:
            U_sv, s_vals, Vt = svd(mat, full_matrices=False)
        except Exception:
            break
        
        tol = 1e-9
        r = int(np.sum(s_vals > tol))
        
        if r >= sub_rank:
            # No more dependencies found with current tolerance
            # Try slightly larger tolerance to see if near-dependent columns exist
            for tol_try in [1e-8, 1e-7, 1e-6]:
                r2 = int(np.sum(s_vals > tol_try))
                if r2 < sub_rank:
                    # Find the least significant direction
                    # Remove the column with smallest contribution
                    null_basis = Vt[r2:]
                    for row in null_basis:
                        for nc, orig_c in enumerate(remaining):
                            if abs(row[nc]) > tol_try:
                                if orig_c not in remaining:
                                    continue
                                remaining.remove(orig_c)
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
            else:
                break
        else:
            # There are dependencies. Find a column to remove.
            # The null space dimension is sub_rank - r
            null_basis = Vt[r:]
            removed = False
            for row in null_basis:
                for nc, orig_c in enumerate(remaining):
                    if abs(row[nc]) > tol:
                        remaining.remove(orig_c)
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break
    
    if len(remaining) < current_rank:
        # Rebuild with remaining columns
        new_U = U_arr[:, remaining].tolist()
        new_V = V_arr[:, remaining].tolist()
        new_W = W_arr[:, remaining].tolist()
        return {"U": new_U, "V": new_V, "W": new_W}
    
    return None

def _try_gaussian_elimination_reduction(U, V, W, n, m, p):
    """
    Alternative: construct the full bilinear tensor, then find its rank
    via random projection + rank estimation, then attempt to project.
    This is a heuristic and may not reduce rank but can confirm minimal rank.
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    total_dim = n * m * p
    
    # Build T: for each (i,j,k), the vector of length rank
    T = np.zeros((total_dim, rank), dtype=float)
    for idx in range(total_dim):
        i = idx // (m * p)
        rem = idx % (m * p)
        j = rem // p
        k = rem % p
        T[idx] = U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i]
    
    from scipy.linalg import svd
    try:
        _, s_vals, _ = svd(T, full_matrices=False)
    except Exception:
        return None
    
    actual_rank = int(np.sum(s_vals > 1e-9))
    return actual_rank

def _construct_strassen_with_shared_terms():
    """
    Attempt to share Strassen terms across adjacent blocks.
    
    Key insight: adjacent A-blocks (a_blk=0 and a_blk=1) share the same row indices.
    Adjacent B-blocks share column structure. We check if two columns
    (from different blocks) have the same "pattern" up to a constant.
    
    For this to work, we'd need:
    U[:,c1] * alpha = U[:,c2] for some alpha,
    V[:,c1] * beta = V[:,c2] for some beta,  
    W[:,c1] * gamma = W[:,c2] for some gamma,
    with alpha * beta * gamma = 1.
    
    This is unlikely for Strassen, but we verify.
    """
    n, m, p = DIMENSIONS
    result = _construct_strassen_56()
    U_arr = np.array(result["U"], dtype=float)
    V_arr = np.array(result["V"], dtype=float)
    W_arr = np.array(result["W"], dtype=float)
    rank = U_arr.shape[1]
    
    # For each pair of columns, check if they are proportional across all three matrices
    # with the product of ratios equal to 1
    removable = set()
    
    for c1 in range(rank):
        if c1 in removable:
            continue
        for c2 in range(c1 + 1, rank):
            if c2 in removable:
                continue
            # Check if U[:,c1] and U[:,c2] are proportional
            u1 = U_arr[:, c1]
            u2 = U_arr[:, c2]
            v1 = V_arr[:, c1]
            v2 = V_arr[:, c2]
            w1 = W_arr[:, c1]
            w2 = W_arr[:, c2]
            
            # Find ratio from U
            nz_u1 = np.nonzero(u1)[0]
            nz_u2 = np.nonzero(u2)[0]
            if len(nz_u1) == 0 or len(nz_u2) == 0:
                continue
            
            # Check if u1 * alpha = u2
            idx0 = nz_u1[0]
            if abs(u2[idx0]) < 1e-12:
                continue
            alpha = u1[idx0] / u2[idx0]
            
            # Verify proportionality on all non-zero entries of both
            valid = True
            for i in range(len(u1)):
                if abs(u1[i] * alpha - u2[i]) > 1e-9:
                    valid = False
                    break
            if not valid:
                continue
            
            # Find ratio from V
            nz_v1 = np.nonzero(v1)[0]
            nz_v2 = np.nonzero(v2)[0]
            if len(nz_v1) == 0 or len(nz_v2) == 0:
                continue
            idx0 = nz_v1[0]
            if abs(v2[idx0]) < 1e-12:
                continue
            beta = v1[idx0] / v2[idx0]
            valid = True
            for i in range(len(v1)):
                if abs(v1[i] * beta - v2[i]) > 1e-9:
                    valid = False
                    break
            if not valid:
                continue
            
            # Find ratio from W
            nz_w1 = np.nonzero(w1)[0]
            nz_w2 = np.nonzero(w2)[0]
            if len(nz_w1) == 0 or len(nz_w2) == 0:
                continue
            idx0 = nz_w1[0]
            if abs(w2[idx0]) < 1e-12:
                continue
            gamma = w1[idx0] / w2[idx0]
            valid = True
            for i in range(len(w1)):
                if abs(w1[i] * gamma - w2[i]) > 1e-9:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check alpha * beta * gamma ≈ 1
            if abs(alpha * beta * gamma - 1.0) < 1e-9:
                # Column c2 can be absorbed: scale column c1 by appropriate amount
                # Actually, we can replace c1's contribution: 
                # T contribution from c2 = (U2 V2 W2) = (U1/alpha)(V1/beta)(W1/gamma) = U1 V1 W1 / (alpha*beta*gamma)
                # = U1 V1 W1 since alpha*beta*gamma = 1
                # So c2 is exactly equivalent to c1 in the sum!
                # We can remove c2 (it's a duplicate)
                removable.add(c2)
    
    if removable:
        keep = [c for c in range(rank) if c not in removable]
        new_U = U_arr[:, keep].tolist()
        new_V = V_arr[:, keep].tolist()
        new_W = W_arr[:, keep].tolist()
        return {"U": new_U, "V": new_V, "W": new_W}
    
    return None

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Start with the rank-56 Strassen construction
    base_result = _construct_strassen_56()
    if _verify(base_result["U"], base_result["V"], base_result["W"], n, m, p):
        best = base_result
    else:
        return _schoolbook_decomposition()
    
    # Try to reduce via null-space analysis
    if time.time() < deadline - 30:
        try:
            reduced = _try_rank_reduction_via_nullspace(
                base_result["U"], base_result["V"], base_result["W"], n, m, p, max_iter=300
            )
            if reduced is not None:
                if _verify(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                    best = reduced
        except Exception:
            pass
    
    # Try to find duplicate/proportional columns
    if time.time() < deadline - 20:
        try:
            dup_result = _construct_strassen_with_shared_terms()
            if dup_result is not None:
                if _verify(dup_result["U"], dup_result["V"], dup_result["W"], n, m, p):
                    best = dup_result
        except Exception:
            pass
    
    # Verify best result
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    return base_result if _verify(base_result["U"], base_result["V"], base_result["W"], n, m, p) \
        else _schoolbook_decomposition()

# EVOLVE-BLOCK-END
