import numpy as np
import time

def construct():
    """
    Rank-reduced decomposition for 4x4 * 4x8 matrix multiplication.
    Uses SVD on the full 4x4x16 tensor to find a low-rank decomposition,
    then restricts to the first 32 output positions for the 4x4x8 problem.
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition as fallback
    U_base, V_base, W_base = _construct_rank98(n, m, p)
    
    # Verify the base decomposition
    if not _verify_decomposition(U_base, V_base, W_base, n, m, p):
        U_base, V_base, W_base = _construct_naive(n, m, p)
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    # Try SVD-based reduction on the full 4x4x16 tensor
    best_result = (U_base, V_base, W_base)
    best_rank = U_base.shape[1]
    
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 10:
        reduced = _reduce_via_full_tensor_svd(n, m, p, deadline - start_time)
        if reduced is not None:
            U_r, V_r, W_r = reduced
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                if U_r.shape[1] < best_rank:
                    best_result = (U_r, V_r, W_r)
                    best_rank = U_r.shape[1]
    
    # Try iterative column elimination on the best result
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 15:
        further_reduced = _iterative_column_elimination(best_result[0], best_result[1], best_result[2], n, m, p, deadline - start_time)
        if further_reduced is not None:
            U_f, V_f, W_f = further_reduced
            if _verify_decomposition(U_f, V_f, W_f, n, m, p):
                if U_f.shape[1] < best_rank:
                    best_result = (U_f, V_f, W_f)
                    best_rank = U_f.shape[1]
    
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}


def _construct_rank98(n, m, p):
    """Rank-98 decomposition using Kronecker product of Strassen levels."""
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return U, V, W


def _reduce_via_full_tensor_svd(n, m, p, time_budget):
    """
    Embed 4x4x8 into 4x4x16 (zero-pad output), find low-rank SVD of the
    full tensor, then restrict back to the first p*n=32 output positions.
    """
    start = time.time()
    n, m, p = 4, 4, 8
    p_full = 16  # embed into 4x4x16
    
    # Build the full 4x4x16 multiplication tensor as a matrix (flattened)
    # Tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Shape: (n*m) x (m*p_full) x (p_full*n) -> flatten to (n*m*p_full, rank)
    
    # The full 4x4x16 tensor has dimension n*m*p_full = 4*4*16 = 256
    # We need to find its rank and a decomposition
    
    # Build the tensor as a 3D array, then flatten
    T = np.zeros((n, m, p_full), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p_full):
                T[i, j, k] = 1.0  # This is the identity tensor for matmul
    
    # Flatten to a matrix: reshape to (n*m, m*p_full, p_full*n) then to (n*m, m*p_full*p_full*n)
    # Actually, the tensor is T[i,j,k] and we want to decompose it as sum_t U[i,j,t]*V[j,k,t]*W[k,i,t]
    # The tensor entries are: T[i,j,k] = delta(i,j,k) for matmul
    # For matmul: C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all valid (i,j,k)
    # But this is the identity tensor, which has rank n*m*p for naive
    
    # Better approach: build the tensor from the known rank-98 decomposition
    # and then find a lower-rank representation via SVD on the flattened tensor
    
    # Let's build the 4x4x16 tensor from two 4x4x4 Strassen blocks
    U4, V4, W4 = _get_4x4x4_strassen()
    
    # Full 4x4x16: first 4 output cols use block 1, next 4 use block 2, etc.
    # Actually for 4x4x16, we need 4 independent 4x4x4 blocks
    rank_full = 49 * 4  # 196 for naive block construction
    
    U_full = np.zeros((n*m, rank_full), dtype=float)
    V_full = np.zeros((m*p_full, rank_full), dtype=float)
    W_full = np.zeros((p_full*n, rank_full), dtype=float)
    
    for block in range(4):
        U_full[:, block*49:(block+1)*49] = U4
        for j in range(m):
            for k in range(4):
                V_full[j*p_full + block*4 + k, block*49:(block+1)*49] = V4[j*4+k]
        for k in range(4):
            for i in range(n):
                W_full[(block*4+k)*n + i, block*49:(block+1)*49] = W4[k*n+i]
    
    # Now we have a rank-196 decomposition of 4x4x16
    # Build the tensor matrix: for each output position (k,i), the tensor slice is
    # T_ki[i,j] = sum_t U[i,j,t] * V[j,block*k+local_k,t] * W[k,i,t]
    # Flatten all (i,j,k) into a single vector of length n*m*p_full = 256
    
    i_idx = np.repeat(np.arange(n), m * p_full)
    j_idx = np.tile(np.repeat(np.arange(m), p_full), n)
    k_idx = np.tile(np.arange(p_full), n * m)
    
    u_flat_idx = i_idx * m + j_idx
    v_flat_idx = j_idx * p_full + k_idx
    w_flat_idx = k_idx * n + i_idx
    
    U_vals = U_full[u_flat_idx, :]  # (256, 196)
    V_vals = V_full[v_flat_idx, :]  # (256, 196)
    W_vals = W_full[w_flat_idx, :]  # (256, 196)
    
    # Column tensors: col_tensors[t, idx] = U_vals[idx,t] * V_vals[idx,t] * W_vals[idx,t]
    col_tensors = (U_vals * V_vals * W_vals).T  # (196, 256)
    
    if time.time() - start > time_budget * 0.3:
        return None
    
    # SVD to find the rank
    try:
        U_svd, s, Vt = np.linalg.svd(col_tensors, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    tol = 1e-10
    num_rank = int(np.sum(s > tol))
    
    if num_rank >= rank_full:
        return None
    
    # Get coordinates in the SVD basis
    coords = U_svd[:, :num_rank] * s[:num_rank]  # (196, num_rank)
    
    if time.time() - start > time_budget * 0.5:
        return None
    
    # Find a basis subset
    basis_indices = _find_basis_subset(coords, num_rank)
    if basis_indices is None:
        return None
    
    # Compute scaling factors
    basis_coords = coords[basis_indices, :]
    try:
        inv_basis = np.linalg.inv(basis_coords)
    except np.linalg.LinAlgError:
        return None
    
    all_alpha = coords @ inv_basis  # (196, num_rank)
    S = all_alpha.sum(axis=0)  # (num_rank,)
    
    if np.any(np.abs(S) < 1e-12):
        return None
    
    # Build reduced decomposition for 4x4x16
    U_red = U_full[:, basis_indices].copy()
    V_red = V_full[:, basis_indices].copy()
    W_red = W_full[:, basis_indices].copy()
    U_red *= S[np.newaxis, :]
    
    # Now restrict to 4x4x8: keep only output positions k=0..7
    # W_red has shape (16*4, num_rank) = (64, num_rank)
    # We need W for k=0..7: rows 0..31
    W_8 = W_red[:p*n, :]  # (32, num_rank)
    
    # V_red has shape (4*16, num_rank) = (64, num_rank)
    # We need V for k=0..7: for each j, keep rows j*16+0 .. j*16+7
    V_8 = np.zeros((m*p, num_rank), dtype=float)
    for j in range(m):
        V_8[j*p:(j+1)*p, :] = V_red[j*p_full:(j+1)*4, :]
    
    # U stays the same
    U_8 = U_red  # (16, num_rank)
    
    if time.time() - start > time_budget * 0.8:
        return None
    
    return U_8, V_8, W_8


def _get_4x4x4_strassen():
    """Get the 4x4x4 rank-49 Strassen decomposition."""
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    return U4, V4, W4


def _find_basis_subset(coords, num_rank):
    """Find a subset of rows that forms a basis for the row space."""
    rank = coords.shape[0]
    
    # Use QR with column pivoting on coords.T
    try:
        Q, R, piv = np.linalg.qr(coords.T, mode='complete', pivoting=True)
    except np.linalg.LinAlgError:
        return None
    
    basis_indices = sorted(piv[:num_rank])
    
    # Verify independence
    basis_coords = coords[basis_indices, :]
    try:
        det_val = np.linalg.det(basis_coords)
    except np.linalg.LinAlgError:
        return None
    
    if abs(det_val) < 1e-10:
        # Try greedy selection
        return _greedy_basis_selection(coords, num_rank)
    
    return basis_indices


def _iterative_column_elimination(U, V, W, n, m, p, time_budget):
    """
    Iteratively try to eliminate columns by expressing them as linear combinations
    of remaining columns with appropriate scaling.
    """
    start = time.time()
    rank = U.shape[1]
    dim = n * m * p
    
    # Build column tensors
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_flat_idx = i_idx * m + j_idx
    v_flat_idx = j_idx * p + k_idx
    w_flat_idx = k_idx * n + i_idx
    
    U_vals = U[u_flat_idx, :]
    V_vals = V[v_flat_idx, :]
    W_vals = W[w_flat_idx, :]
    col_tensors = (U_vals * V_vals * W_vals).T  # (rank, dim)
    
    # Compute SVD
    try:
        U_svd, s, Vt = np.linalg.svd(col_tensors, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    tol = 1e-10
    num_rank = int(np.sum(s > tol))
    
    if num_rank >= rank:
        return None
    
    coords = U_svd[:, :num_rank] * s[:num_rank]
    
    # Find basis
    basis_indices = _find_basis_subset(coords, num_rank)
    if basis_indices is None:
        return None
    
    basis_coords = coords[basis_indices, :]
    try:
        inv_basis = np.linalg.inv(basis_coords)
    except np.linalg.LinAlgError:
        return None
    
    all_alpha = coords @ inv_basis
    S = all_alpha.sum(axis=0)
    
    if np.any(np.abs(S) < 1e-12):
        return None
    
    U_r = U[:, basis_indices].copy()
    V_r = V[:, basis_indices].copy()
    W_r = W[:, basis_indices].copy()
    U_r *= S[np.newaxis, :]
    
    return U_r, V_r, W_r


def _construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Test with random matrices
    rng = np.random.default_rng(42)
    for trial in range(3):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        M = np.zeros(rank)
        for t in range(rank):
            u_val = np.dot(U[:, t], A_flat)
            v_val = np.dot(V[:, t], B_flat)
            M[t] = u_val * v_val
        
        C_computed = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                C_computed[i, k] = np.dot(W[k*n+i, :], M)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _greedy_basis_selection(coords, num_rank):
    """Greedy selection of linearly independent rows from coords."""
    rank = coords.shape[0]
    selected = []
    
    for t in range(rank):
        if len(selected) >= num_rank:
            break
        # Check if coords[t] is in the span of already selected rows
        if not selected:
            selected.append(t)
        else:
            A = coords[selected, :]  # (k, num_rank)
            b = coords[t, :]
            try:
                x, residuals, _, _ = np.linalg.lstsq(A.T, b, rcond=1e-12)
                if residuals.size > 0 and residuals[0] < 1e-10:
                    continue  # dependent, skip
            except np.linalg.LinAlgError:
                continue
            selected.append(t)
    
    if len(selected) < num_rank:
        return None
    return selected[:num_rank]
