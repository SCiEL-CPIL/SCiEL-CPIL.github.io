<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition with tolerance for floating point."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_true = _build_true_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness with tight tolerance
    if not np.allclose(T_recon, T_true, atol=1e-8):
        return False
    return True

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _try_rank_als(r, n, m, p, T_true, max_iters=2000, seed=42, time_budget=30):
    """Try to find rank-r decomposition using ALS with half-integer rounding."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    best_solution = None
    best_err = float('inf')
    
    for init_seed in range(3):
        if time.time() - start_time > time_budget:
            break
        
        rng = np.random.RandomState(seed + init_seed)
        
        # Initialize with random values
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            
            # Check if we're close enough
            err = np.max(np.abs(R))
            if err < 1e-6:
                # Try rounding to half-integers
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    best_solution = (U_r, V_r, W_r)
                    best_err = err_r
                    return best_solution
                
                # If rounding didn't work, continue with original
                if err < 1e-8:
                    # Keep original values if they work exactly
                    best_solution = (U, V, W)
                    best_err = err
            
            # ALS updates
            # Update U: for each a, solve (V*W^T) U[a,:]^T = T_true[a,:,:] weighted
            # Simpler: gradient-based update
            eta = 0.05
            
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            U += eta * grad_U
            V += eta * grad_V
            W += eta * grad_W
            
            # Periodically check and try rounding
            if it % 50 == 0 and err < 0.1:
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    best_solution = (U_r, V_r, W_r)
                    best_err = err_r
                    return best_solution
    
    return best_solution

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    """
    Constructs the rank-minimal tensor decomposition for 
    multiplying a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T_true = _build_true_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Lower bound is max(nm, mp, pn) = 32
    lower_bound = max(nm, mp, pn)
    
    # Try ranks from lower_bound up to best_rank-1
    # Limit total search time to 100 seconds (leaving 20s safety margin)
    search_start = time.time()
    max_search_time = 100.0
    
    # Try a subset of ranks to stay within time budget
    # Focus on ranks that are "structured" (multiples of small numbers)
    candidate_ranks = []
    for r in range(lower_bound, best_rank):
        candidate_ranks.append(r)
    
    # Sort by likelihood of success (try lower ranks first)
    for r in candidate_ranks:
        if time.time() - search_start > max_search_time:
            break
        
        if r >= best_rank:
            break
        
        # Try to find a rank-r decomposition
        result = _try_rank_als(r, n, m, p, T_true, 
                               max_iters=1000, seed=42, 
                               time_budget=15)
        
        if result is not None:
            U_r, V_r, W_r = result
            # Verify
            if _verify_exact(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                best_rank = r
                # If we found a good solution, we can stop early
                # or continue to try even lower ranks
                if r <= lower_bound + 10:
                    break
    
    return best_result
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition with tolerance for floating point."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_true = _build_true_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness with tight tolerance
    if not np.allclose(T_recon, T_true, atol=1e-8):
        return False
    return True

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) with integer entries.
    The tensor is T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    where a indexes row of first matrix (2x2), b indexes row of second (2x2),
    c indexes col of first / row of output.
    Mapping: a = i*2+j (i=row of A, j=col of A), b = j*2+k (j=row of B, k=col of B),
             c = k*2+i (k=col of B, i=row of A)
    """
    # Strassen's 7 products for 2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # Indexing: A[i,j] -> a = i*2+j, B[j,k] -> b = j*2+k, C[i,k] -> c = k*2+i
    #
    # U[a, r] = coefficient of A[a] in product r
    # V[b, r] = coefficient of B[b] in product r
    # W[c, r] = coefficient of C[c] from product r
    #
    # Products indexed r=0..6:
    # r=0: M1 = (a11+a22)*(b11+b22)
    #   U: a11=1, a22=1 -> U[0,0]=1, U[3,0]=1
    #   V: b11=1, b22=1 -> V[0,0]=1, V[3,0]=1
    #   W: c11=M1, c22=M1 -> W[1,0]=1 (c11: k=0,i=0 -> c=0*2+0=0), W[0,0]=1 (c22: k=1,i=1 -> c=1*2+1=3)
    #   Wait, let me redo: c = k*2 + i
    #   c11: i=0,k=0 -> c=0
    #   c12: i=0,k=1 -> c=2
    #   c21: i=1,k=0 -> c=1
    #   c22: i=1,k=1 -> c=3
    #   c11 = M1 + M4 - M5 + M7 -> W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
    #   c12 = M3 + M5 -> W[2,2]=1, W[2,4]=1
    #   c21 = M2 + M4 -> W[1,1]=1, W[1,3]=1
    #   c22 = M1 - M2 + M3 + M6 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1

    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # r=0: M1 = (a11+a22)*(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # c11
    W[3, 0] = 1  # c22

    # r=1: M2 = (a21+a22)*b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # c21
    W[3, 1] = -1 # c22

    # r=2: M3 = a11*(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1  # c12
    W[3, 2] = 1  # c22

    # r=3: M4 = a22*(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # c11
    W[1, 3] = 1  # c21

    # r=4: M5 = (a11+a12)*b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # c11
    W[2, 4] = 1  # c12

    # r=5: M6 = (a21-a11)*(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # c22

    # r=6: M7 = (a12-a22)*(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # c11

    return U, V, W

def _embed_strassen(n, m, p, n_sub, m_sub, p_sub):
    """
    Embed the Strassen 2x2x2 decomposition into the n x m x p tensor.
    We pick specific rows/cols to map the 2x2 sub-problem into the larger problem.
    For 3x4x8: n=3, m=4, p=8.
    We map:
      - 2 rows of A (out of 3)
      - 2 cols of A / 2 rows of B (out of 4)
      - 2 cols of B (out of 8)
    
    The embedding uses identity subspaces: we pick specific indices.
    Returns (U, V, W) of shapes (n*m, 7), (m*p, 7), (p*n, 7)
    plus the residual tensor R = T_true - T_embedded.
    """
    T_true = _build_true_tensor(n, m, p)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Choose which indices to use for the 2x2 sub-problem
    # For n=3: use rows i=0,1 of A
    # For m=4: use cols j=0,1 of A (= rows of B)
    # For p=8: use cols k=0,1 of B
    
    i_map = [0, 1]  # rows of A
    j_map = [0, 1]  # cols of A / rows of B
    k_map = [0, 1]  # cols of B
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = np.zeros((nm, 7), dtype=float)
    V = np.zeros((mp, 7), dtype=float)
    W = np.zeros((pn, 7), dtype=float)
    
    for r in range(7):
        for a_sub in range(4):
            i_sub, j_sub = a_sub // 2, a_sub % 2
            i = i_map[i_sub]
            j = j_map[j_sub]
            U[i * m + j, r] = U_s[a_sub, r]
        
        for b_sub in range(4):
            j_sub, k_sub = b_sub // 2, b_sub % 2
            j = j_map[j_sub]
            k = k_map[k_sub]
            V[j * p + k, r] = V_s[b_sub, r]
        
        for c_sub in range(4):
            k_sub, i_sub = c_sub // 2, c_sub % 2
            k = k_map[k_sub]
            i = i_map[i_sub]
            W[k * n + i, r] = W_s[c_sub, r]
    
    T_embed = np.einsum('ar,br,cr->abc', U, V, W)
    R = T_true - T_embed
    
    return U, V, W, R

def _rank1_decompose_residual(R, max_rank, n, m, p, time_budget=30, seed=42):
    """
    Try to decompose residual tensor R into at most max_rank rank-1 terms.
    Uses iterative closest rank-1 approximation (greedy).
    Returns list of (u, v, w) vectors or None if not successful.
    """
    start_time = time.time()
    nm, mp, pn = R.shape
    R_work = R.copy()
    terms = []
    
    for step in range(max_rank):
        if time.time() - start_time > time_budget:
            return None
        
        # Find best rank-1 approximation to R_work
        # Use SVD on matricizations
        # Flatten R_work to a matrix and do SVD
        # Try multiple matricizations and pick best
        
        best_u, best_v, best_w = None, None, None
        best_err = float('inf')
        
        for mat_type in range(3):
            if mat_type == 0:
                # Matricize as (nm, mp*pn)
                M = R_work.reshape(nm, mp * pn)
                try:
                    U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
                    u = U_svd[:, 0]
                    vw = Vt[0].reshape(mp, pn)
                    # SVD of vw
                    U2, S2, V2t = np.linalg.svd(vw, full_matrices=False)
                    v = U2[:, 0]
                    w = V2t[0]
                    scale = S[0] * S2[0]
                    u = u * scale
                    v = v
                    w = w
                except:
                    continue
            elif mat_type == 1:
                # Matricize as (mp, nm*pn)
                M = R_work.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
                    v = U_svd[:, 0]
                    uw = Vt[0].reshape(nm, pn)
                    U2, S2, V2t = np.linalg.svd(uw, full_matrices=False)
                    u = U2[:, 0]
                    w = V2t[0]
                    scale = S[0] * S2[0]
                    v = v * scale
                    u = u
                    w = w
                except:
                    continue
            else:
                # Matricize as (pn, nm*mp)
                M = R_work.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
                    w = U_svd[:, 0]
                    uv = Vt[0].reshape(nm, mp)
                    U2, S2, V2t = np.linalg.svd(uv, full_matrices=False)
                    u = U2[:, 0]
                    v = V2t[0]
                    scale = S[0] * S2[0]
                    w = w * scale
                    u = u
                    v = v
                except:
                    continue
            
            if u is None:
                continue
            
            # Compute error after subtracting this rank-1 term
            R_new = R_work - np.einsum('a,b,c->abc', u, v, w)
            err = np.max(np.abs(R_new))
            
            if err < best_err:
                best_err = err
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None:
            return None
        
        terms.append((best_u, best_v, best_w))
        R_work = R_work - np.einsum('a,b,c->abc', best_u, best_v, best_w)
        
        if np.max(np.abs(R_work)) < 1e-10:
            break
    
    if np.max(np.abs(R_work)) < 1e-8:
        return terms
    return None

def _polish_decomposition(U, V, W, T_true, n, m, p, time_budget=30, tol=1e-10):
    """
    High-precision polish: given an approximate decomposition, refine it
    using alternating least squares with tight tolerance.
    Returns refined (U, V, W) or None.
    """
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(500):
        if time.time() - start_time > time_budget:
            return None
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T_true - T_recon
        err = np.max(np.abs(R))
        
        if err < tol:
            return (U, V, W)
        
        # ALS: update each factor
        # Update U: for each a, U[a,:] = (V @ W^T)^{-1} @ R[a,:,:].flatten()
        # Actually: R[a,:,:] = U[a,:]^T * (V * W) where (V*W)[r, b, c] = V[b,r]*W[c,r]
        # So R[a,:,:].flatten() = (V*W^T) @ U[a,:]
        # Let M = V @ W^T (shape r x r... no)
        # R[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed a: R[a,:,:] = sum_r U[a,r] outer(V[:,r], W[:,r])
        # Vectorize: R[a].flatten() = sum_r U[a,r] (V[:,r] * W[:,r]).flatten()
        # So R[a].flatten() = B @ U[a,:] where B[b,c,r] = V[b,r]*W[c,r] -> B as (mp*pn, r)
        
        B = (V * W[None, :, :]).reshape(mp, pn, r).transpose(2, 0, 1).reshape(r, mp * pn)
        # B has shape (r, mp*pn), we want B^T (mp*pn, r)
        Bt = B.T
        
        for a in range(nm):
            target = R[a].flatten()
            try:
                U[a, :] = np.linalg.lstsq(Bt, target, rcond=None)[0]
            except:
                pass
        
        # Similarly for V
        A = (U * W[None, :, :]).reshape(nm, pn, r).transpose(2, 0, 1).reshape(r, nm * pn)
        At = A.T
        for b in range(mp):
            target = R[:, b, :].transpose(1, 0).flatten()
            # R[:,b,:] has shape (nm, pn), flatten to (nm*pn,)
            # But we need R[a,b,c] for fixed b: sum_r U[a,r] V[b,r] W[c,r]
            # For fixed b: R[:,b,:].flatten() = sum_r V[b,r] (U[:,r] * W[:,r]).flatten()
            # = (U*W).reshape(nm,pn,r).transpose(2,0,1).reshape(r, nm*pn) @ V[b,:]
            target = R[:, b, :].reshape(nm, pn).transpose(1, 0).flatten()
            # Actually let's just do it properly
            target = R[:, b, :].flatten()
            try:
                V[b, :] = np.linalg.lstsq(At, target, rcond=None)[0]
            except:
                pass
        
        # Similarly for W
        C = (U * V[None, :, :]).reshape(nm, mp, r).transpose(2, 0, 1).reshape(r, nm * mp)
        Ct = C.T
        for c in range(pn):
            target = R[:, :, c].flatten()
            try:
                W[c, :] = np.linalg.lstsq(Ct, target, rcond=None)[0]
            except:
                pass
    
    # Final check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T_true))
    if err < tol:
        return (U, V, W)
    return None

def _try_rank_als(r, n, m, p, T_true, max_iters=2000, seed=42, time_budget=30):
    """Try to find rank-r decomposition using ALS with half-integer rounding."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    best_solution = None
    best_err = float('inf')
    
    for init_seed in range(3):
        if time.time() - start_time > time_budget:
            break
        
        rng = np.random.RandomState(seed + init_seed)
        
        # Initialize with random values
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            
            # Check if we're close enough
            err = np.max(np.abs(R))
            if err < 1e-6:
                # Try rounding to half-integers
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    best_solution = (U_r, V_r, W_r)
                    best_err = err_r
                    return best_solution
                
                # If rounding didn't work, continue with original
                if err < 1e-8:
                    # Keep original values if they work exactly
                    best_solution = (U, V, W)
                    best_err = err
            
            # Gradient-based update
            eta = 0.05
            
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            U += eta * grad_U
            V += eta * grad_V
            W += eta * grad_W
            
            # Periodically check and try rounding
            if it % 50 == 0 and err < 0.1:
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    best_solution = (U_r, V_r, W_r)
                    best_err = err_r
                    return best_solution
    
    return best_solution

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_embedding(n, m, p, time_budget=60):
    """
    Try to find a low-rank decomposition by embedding Strassen's 2x2x2
    decomposition and then greedily adding rank-1 terms to cancel the residual.
    """
    start_time = time.time()
    
    U, V, W, R = _embed_strassen(n, m, p)
    
    residual_norm = np.max(np.abs(R))
    
    # Try to decompose the residual into rank-1 terms
    # The residual has structure: it only involves indices outside the 2x2x2 core
    # Let's try up to some number of additional terms
    max_extra = 30
    
    for max_terms in range(1, max_extra + 1):
        if time.time() - start_time > time_budget:
            break
        
        terms = _rank1_decompose_residual(R, max_terms, n, m, p, 
                                           time_budget=time_budget - (time.time() - start_time),
                                           seed=42)
        
        if terms is not None:
            # Combine Strassen + residual terms
            total_rank = 7 + len(terms)
            U_full = np.zeros((n*m, total_rank), dtype=float)
            V_full = np.zeros((m*p, total_rank), dtype=float)
            W_full = np.zeros((p*n, total_rank), dtype=float)
            
            U_full[:, :7] = U
            V_full[:, :7] = V
            W_full[:, :7] = W
            
            for idx, (u, v, w) in enumerate(terms):
                col = 7 + idx
                U_full[:, col] = u
                V_full[:, col] = v
                W_full[:, col] = w
            
            # Verify
            T_recon = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
            T_true = _build_true_tensor(n, m, p)
            err = np.max(np.abs(T_recon - T_true))
            
            if err < 1e-8:
                # Try to round to half-integers
                U_r = _round_to_half_integer(U_full)
                V_r = _round_to_half_integer(V_full)
                W_r = _round_to_half_integer(W_full)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist(), "rank": total_rank}
                else:
                    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist(), "rank": total_rank}
    
    return None

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 
    multiplying a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T_true = _build_true_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Wall-clock deadline guard
    search_start = time.time()
    deadline = search_start + HARD_LIMIT - SAFETY_MARGIN
    
    # Strategy 1: Strassen embedding + residual rank-1 decomposition
    if time.time() < deadline:
        strassen_result = _try_strassen_embedding(n, m, p, 
                                                   time_budget=min(60.0, deadline - time.time()))
        if strassen_result is not None:
            if strassen_result["rank"] < best_rank:
                best_result = {
                    "U": strassen_result["U"],
                    "V": strassen_result["V"],
                    "W": strassen_result["W"]
                }
                best_rank = strassen_result["rank"]
    
    # Strategy 2: Try direct ALS for lower ranks
    lower_bound = max(nm, mp, pn)
    
    if time.time() < deadline:
        # Try a few candidate ranks
        for r in range(lower_bound, min(best_rank, lower_bound + 20)):
            if time.time() > deadline:
                break
            
            remaining_time = deadline - time.time()
            if remaining_time < 5:
                break
            
            result = _try_rank_als(r, n, m, p, T_true, 
                                   max_iters=500, seed=42, 
                                   time_budget=min(15.0, remaining_time - 2))
            
            if result is not None:
                U_r, V_r, W_r = result
                if _verify_exact(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    if r < best_rank:
                        best_result = {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        best_rank = r
                        if r <= lower_bound + 5:
                            break
    
    # Final safety check
    if time.time() > deadline - 5:
        # Return best result immediately
        pass
    
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
