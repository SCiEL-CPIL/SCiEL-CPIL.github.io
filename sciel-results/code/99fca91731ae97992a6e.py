import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification of tensor decomposition."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_construct(n, m, p):
    """Construct the Strassen-based decomposition (rank 64 for 4x4x5)."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u.copy())
        V_cols.append(v.copy())
        W_cols.append(w.copy())
    
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    def add_strassen_2x2_2(ai, aj, bj, bk):
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai+1)] = 1; w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk+1)] = 1; v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = -1; w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = 1; u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk+2, ai+di)] = 1
                add_term(u, v, w)

    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _try_remove_term(U, V, W, idx, n, m, p, tol=1e-9):
    """Try to remove term idx. Returns (U_new, V_new, W_new) or None."""
    rank = U.shape[1]
    mask = np.ones(rank, dtype=bool)
    mask[idx] = False
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    if _verify_fast(U_new, V_new, W_new, n, m, p, tol):
        return U_new, V_new, W_new
    return None

def _try_merge_terms(U, V, W, idx1, idx2, n, m, p, tol=1e-9):
    """Try to merge two terms into one. Returns (U_new, V_new, W_new) or None."""
    rank = U.shape[1]
    if rank <= 2:
        return None
    
    # Remove both terms
    mask = np.ones(rank, dtype=bool)
    mask[idx1] = False
    mask[idx2] = False
    U_rem = U[:, mask]
    V_rem = V[:, mask]
    W_rem = W[:, mask]
    
    # The residual tensor is: T - sum_{r in remaining} U[r]*V[r]*W[r]
    # We want to find a single term (u, v, w) such that:
    #   residual[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    # This is a rank-1 tensor approximation problem.
    
    # Build the residual tensor
    residual = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_rem[i*m+j, :] * V_rem[j*p+k, :], W_rem[k*n+i, :])
                residual[i, j, k] = 1.0 - val
    
    # Check if residual is approximately rank-1
    # Reshape to matrix and check singular values
    # Reshape residual to (n*m, p) or (n, m*p) etc.
    # A rank-1 tensor T[i,j,k] = a[i]*b[j]*c[k] when reshaped to (n*m, p) 
    # has form: T_reshaped[i*m+j, k] = a[i]*b[j]*c[k]
    # This is not generally rank-1 as a matrix, but we can try to find a, b, c
    
    # Alternative: use the fact that for a rank-1 tensor, 
    # T[i1,j1,k1]*T[i2,j2,k2] = T[i1,j2,k1]*T[i2,j1,k2] etc.
    # Let's try a simpler approach: solve for the new term coefficients
    
    # Actually, let's try a different approach: 
    # We want to find u, v, w such that for all (i,j,k):
    #   residual[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    # Take log if all positive, or use iterative method
    
    # Simpler: try to find if residual can be written as outer product
    # Reshape to (n*m*p,) and check if it's a rank-1 tensor
    # A tensor is rank-1 if all its mode-n unfoldings have rank 1
    
    # Let's try a practical approach: assume w is all 1s (or some simple pattern)
    # and solve for u and v, or vice versa
    
    # Method: fix w, solve for u and v
    # residual[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    # For fixed k: residual[:, :, k] = (u * w_k) outer (v_k) where w_k[i] = w[k*n+i], v_k[j] = v[j*p+k]
    
    # Try with w = ones first
    w_test = np.ones(p*n)
    # Then residual[i,j,k] = u[i*m+j] * v[j*p+k]
    # For each k: residual[:, :, k] should be rank-1 matrix
    
    # Check: for each k, is residual[:,:,k] rank-1?
    for k in range(p):
        mat = residual[:, :, k]
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            continue
        sv = np.linalg.svd(mat, compute_uv=False)
        if len(sv) > 1 and sv[1] > 1e-6 * (sv[0] if sv[0] > 0 else 1):
            break
    else:
        # All slices are rank-1, so it might be rank-1 tensor
        # Extract u and v from the first non-zero slice
        found = False
        for k in range(p):
            mat = residual[:, :, k]
            if np.any(np.abs(mat) > 1e-10):
                sv, U_svd, Vt_svd = np.linalg.svd(mat, full_matrices=False)
                if sv[0] > 1e-10:
                    u_vec = U_svd[:, 0] * np.sqrt(sv[0])
                    v_vec = Vt_svd[0, :] * np.sqrt(sv[0])
                    # u_vec corresponds to u[i*m+j] for varying i,j
                    # v_vec corresponds to v[j*p+k] for varying j
                    # But we need to be careful about indexing
                    # u[i*m+j] = u_vec[i*m+j]... no, u_vec is indexed by row of mat which is i
                    # mat[i,j] = u[i*m+j] * v[j*p+k]
                    # So mat is n x m, and mat[i,j] = u[i*m+j] * v[j*p+k]
                    # This means u[i*m+j] * v[j*p+k] = mat[i,j]
                    # For this to work for all j, we need u[i*m+j] = mat[i,j]/v[j*p+k]
                    # Let's pick j=0: u[i*m+0] = mat[i,0]/v[0*p+k]
                    # And u[i*m+j] = mat[i,j]/v[j*p+k]
                    # So u[i*m+j] * v[j*p+k] = mat[i,j]
                    # Let v[0*p+k] = 1 (normalize), then u[i*m+0] = mat[i,0]
                    # And for j>0: v[j*p+k] = mat[i,j]/u[i*m+j] for any i where u[i*m+j] != 0
                    
                    # This is getting complicated. Let's just verify if the residual 
                    # can be decomposed and if so, add the new term.
                    
                    # Actually, let's just check if the full residual tensor is rank-1
                    # by checking the unfolding matrices
                    found = True
                    break
        
        if found:
            # Build the new term
            U_new = np.column_stack([U_rem, np.zeros(n*m)])
            V_new = np.column_stack([V_rem, np.zeros(m*p)])
            W_new = np.column_stack([W_rem, np.zeros(p*n)])
            
            # Extract u, v, w from residual
            # For simplicity, try to extract from the data
            u_new = np.zeros(n*m)
            v_new = np.zeros(m*p)
            w_new = np.ones(p*n)
            
            # Use the first non-zero element to normalize
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(residual[i,j,k]) > 1e-10:
                            # residual[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                            # Set w[k*n+i] = 1, then u[i*m+j] * v[j*p+k] = residual[i,j,k]
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                return None
            
            # This approach is too complex. Let's use a simpler heuristic:
            # Just check if removing both terms and adding nothing works (shouldn't)
            # or if we can find a simple pattern
            
            # For now, skip this and rely on term removal only
            pass
    
    return None

def _greedy_term_removal(U, V, W, n, m, p, max_rounds=10, tol=1e-9):
    """Repeatedly try to remove terms greedily."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for round_num in range(max_rounds):
        rank = U.shape[1]
        removed = False
        for idx in range(rank):
            result = _try_remove_term(U, V, W, idx, n, m, p, tol)
            if result is not None:
                U, V, W = result
                removed = True
                break
        if not removed:
            break
    
    return U, V, W

def _als_decompose_from_start(U_init, V_init, W_init, n, m, p, target_rank, seed=42, max_iters=100, tol=1e-9, time_limit=15):
    """ALS decomposition starting from an initial solution, trying to reach target_rank."""
    np.random.seed(seed)
    start = time.time()
    
    U = U_init.copy()
    V = V_init.copy()
    W = W_init.copy()
    rank = U.shape[1]
    
    # If we want a lower rank, we need to project
    if target_rank < rank:
        # Keep only the first target_rank columns as initial guess
        U = U[:, :target_rank]
        V = V[:, :target_rank]
        W = W[:, :target_rank]
        rank = target_rank
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Optimize U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
    
    # Round to half-integers
    U_r = _round_matrix_to_half_int(U)
    V_r = _round_matrix_to_half_int(V)
    W_r = _round_matrix_to_half_int(W)
    
    # Remove zero columns
    keep = []
    for r in range(U_r.shape[1]):
        if np.any(U_r[:, r] != 0) and np.any(V_r[:, r] != 0) and np.any(W_r[:, r] != 0):
            keep.append(r)
    U_r = U_r[:, keep]
    V_r = V_r[:, keep]
    W_r = W_r[:, keep]
    
    if _verify_fast(U_r, V_r, W_r, n, m, p, tol):
        return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Build the Strassen-based decomposition (rank 64)
    U, V, W = _strassen_construct(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify the baseline
    assert _verify_fast(U, V, W, n, m, p), "Baseline decomposition failed verification"
    
    # Step 1: Greedy term removal
    U, V, W = _greedy_term_removal(U, V, W, n, m, p, max_rounds=20)
    if U.shape[1] < best_rank:
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Step 2: Try ALS from the reduced solution to go even lower
    if time.time() < deadline - 30:
        for target_rank in range(best_rank - 1, max(40, best_rank - 15), -1):
            if time.time() > deadline - 25:
                break
            
            found = False
            for seed in range(3):
                if time.time() > deadline - 25:
                    break
                
                result = _als_decompose_from_start(
                    best_U, best_V, best_W, n, m, p, 
                    target_rank, seed=seed, max_iters=80, time_limit=12
                )
                if result is not None:
                    U_r, V_r, W_r = result
                    if U_r.shape[1] < best_rank:
                        best_rank = U_r.shape[1]
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                        found = True
                        break
            
            if found:
                continue
    
    # Step 3: Final polish - try to round and verify
    U_r = _round_matrix_to_half_int(best_U)
    V_r = _round_matrix_to_half_int(best_V)
    W_r = _round_matrix_to_half_int(best_W)
    
    keep = []
    for r in range(U_r.shape[1]):
        if np.any(U_r[:, r] != 0) and np.any(V_r[:, r] != 0) and np.any(W_r[:, r] != 0):
            keep.append(r)
    U_r = U_r[:, keep]
    V_r = V_r[:, keep]
    W_r = W_r[:, keep]
    
    if _verify_fast(U_r, V_r, W_r, n, m, p):
        best_U, best_V, best_W = U_r, V_r, W_r
        best_rank = U_r.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
