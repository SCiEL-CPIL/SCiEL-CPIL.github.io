<<<<<<< SEARCH
def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to find a lower-rank decomposition by identifying linear dependencies."""
    r = U.shape[1]
    
    # Build the constraint rows
    rows = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    rows.append((i, j, jp, k))
    
    n_constraints = len(rows)
    
    # Try eliminating columns iteratively, starting from most likely redundant
    # Schoolbook columns are most likely redundant
    order = list(range(r - 1, r - 9, -1)) + list(range(r - 9, -1, -1))
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    r_cur = r
    removed = set()
    
    for c0 in order:
        if time.time() > deadline - 5:
            break
        if c0 in removed:
            continue
        
        # Build the system: find alpha such that
        # sum_{c not in removed, c != c0} alpha_c * U_c * V_c * W_c = U_c0 * V_c0 * W_c0
        active_cols = [c for c in range(r_cur) if c not in removed and c != c0]
        if len(active_cols) == 0:
            continue
        
        A = np.zeros((n_constraints, len(active_cols)))
        b = np.zeros(n_constraints)
        
        for col_idx, c in enumerate(active_cols):
            for idx2, (i, j, jp, k) in enumerate(rows):
                A[idx2, col_idx] = U_cur[i * m + j, c] * V_cur[jp * p + k, c] * W_cur[k * n + i, c]
        
        for idx2, (i, j, jp, k) in enumerate(rows):
            b[idx2] = U_cur[i * m + j, c0] * V_cur[jp * p + k, c0] * W_cur[k * n + i, c0]
        
        # Solve least squares
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if the solution is accurate
        if len(residuals) > 0:
            if residuals[0] > 1e-8:
                continue
        
        # Verify the solution by constructing new matrices
        U_new = np.zeros((n * m, r_cur - 1))
        V_new = np.zeros((m * p, r_cur - 1))
        W_new = np.zeros((p * n, r_cur - 1))
        
        col_idx = 0
        for c in range(r_cur):
            if c in removed or c == c0:
                continue
            U_new[:, col_idx] = U_cur[:, c] * alpha[col_idx]
            V_new[:, col_idx] = V_cur[:, c]
            W_new[:, col_idx] = W_cur[:, c]
            col_idx += 1
        
        # Verify
        if _verify(U_new, V_new, W_new, n, m, p):
            U_cur = U_new
            V_cur = V_new
            W_cur = W_new
            r_cur -= 1
            removed.add(c0)
            continue
        
        # Try with different scaling: put alpha in V instead
        U_new = np.zeros((n * m, r_cur - 1))
        V_new = np.zeros((m * p, r_cur - 1))
        W_new = np.zeros((p * n, r_cur - 1))
        
        col_idx = 0
        for c in range(r_cur):
            if c in removed or c == c0:
                continue
            U_new[:, col_idx] = U_cur[:, c]
            V_new[:, col_idx] = V_cur[:, c] * alpha[col_idx]
            W_new[:, col_idx] = W_cur[:, c]
            col_idx += 1
        
        if _verify(U_new, V_new, W_new, n, m, p):
            U_cur = U_new
            V_cur = V_new
            W_cur = W_new
            r_cur -= 1
            removed.add(c0)
            continue
    
    if r_cur < r:
        return U_cur[:, :r_cur], V_cur[:, :r_cur], W_cur[:, :r_cur]
    
    return None, None, None
=======
def _build_structured(n, m, p):
    """Build a structured decomposition exploiting the 2x4x7 structure."""
    dim_A = n * m  # 8
    dim_B = m * p  # 28
    dim_C = p * n  # 14
    
    # Strategy: Use the block structure
    # A = [A0 | A1] where A0, A1 are 2x2
    # B = [B0; B1] where B0, B1 are 2x7
    # C = A0@B0 + A1@B1
    
    # For each 2x2 @ 2x7 product, use a rank-14 decomposition (schoolbook on 2x2 part,
    # but sharing the 7 columns)
    # Actually, let's use a smarter approach.
    
    # For 2x2 @ 2x7:
    # C[i,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # = (A[0,0]+A[1,0]) * B[0,k] + (A[0,1]+A[1,1]) * B[1,k]
    # But that's not right, we need separate i indices.
    
    # Let me use: for the 2x2 @ 2x7 block, the rank is at most 2*7=14 (schoolbook)
    # but can be less with clever tricks.
    
    # Actually, the tensor for 2x2 @ 2x7 has rank exactly 14 (I believe).
    # So two blocks give rank 28. Plus we need to combine them.
    
    # Wait, C = A0@B0 + A1@B1. Each term A0@B0 is a 2x7 matrix that's a linear
    # combination of columns of B0 with coefficients from A0.
    
    # The total tensor is the sum of two tensors, each of rank 14.
    # So total rank <= 28.
    
    # Let's build this rank-28 decomposition.
    r = 28
    U = np.zeros((dim_A, r))
    V = np.zeros((dim_B, r))
    W = np.zeros((dim_C, r))
    
    # Block 0: A0 (rows 0-1, cols 0-1 of A) @ B0 (rows 0-1, all cols of B)
    # A0[i,j] is at A[i, j] for i in 0,1 and j in 0,1
    # B0[j,k] is at B[j, k] for j in 0,1 and k in 0..6
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(7):
                U[i * 4 + j, col] = 1.0  # A[i,j]
                V[j * 7 + k, col] = 1.0  # B[j,k]
                W[k * 2 + i, col] = 1.0  # C[i,k]
                col += 1
    
    # Block 1: A1 (rows 0-1, cols 2-3 of A) @ B1 (rows 2-3, all cols of B)
    # A1[i,j] is at A[i, 2+j] for i in 0,1 and j in 0,1
    # B1[j,k] is at B[2+j, k] for j in 0,1 and k in 0..6
    
    for i in range(2):
        for j in range(2):
            for k in range(7):
                U[i * 4 + 2 + j, col] = 1.0  # A[i, 2+j]
                V[(2 + j) * 7 + k, col] = 1.0  # B[2+j, k]
                W[k * 2 + i, col] = 1.0  # C[i,k]
                col += 1
    
    if col != r:
        return None, None, None
    
    return U, V, W
>>>>>>> REPLACE

Now let me also add a more aggressive compression that tries multiple elimination orders:

