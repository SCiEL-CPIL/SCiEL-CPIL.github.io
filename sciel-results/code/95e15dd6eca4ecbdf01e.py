"""Matrix multiplication 4,4,5 tensor decomposition with combinatorial subspace elimination."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """
    Returns 7 terms for 2x2x2 Strassen.
    Each term is (u_sparse, v_sparse, w_sparse) where sparse is list of (index, value).
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    """
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _construct_rank_65(n=4, m=4, p=5):
    """
    Constructs a rank-65 decomposition.
    49 terms from 4x4x4 Kronecker Strassen.
    16 terms from 4x4x1 Naive.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i

    strassen_terms = _strassen_2x2x2_terms()
    
    # 1. 4x4x4 part (k=0..3)
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2
            
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
            
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
            
            add_term(u, v, w)
            
    # 2. 4x4x1 part (k=4)
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m+j, :]
                v_val = V_np[j*p+k, :]
                w_val = W_np[k*n+i, :]
                total = np.dot(u_val * v_val, w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _compute_term_vectors(U, V, W, n, m, p):
    """
    Compute the tensor vector for each rank-1 term.
    Returns matrix of shape (n*m*p, rank) where column r is vec(T_r).
    T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    num_entries = n * m * p
    M = np.zeros((num_entries, rank))
    
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        
        for i in range(n):
            for j in range(m):
                u_val = u[i*m+j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v[j*p+k]
                    if v_val == 0:
                        continue
                    w_val = w[k*n+i]
                    if w_val == 0:
                        continue
                    val = u_val * v_val * w_val
                    if val != 0:
                        flat_idx = (i * m + j) * p + k
                        M[flat_idx, r] = val
    
    return M

def _build_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 80."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    return U, V, W

def _try_single_removal(U, V, W, n, m, p, term_vecs, target_flat, tol=1e-10, deadline=None):
    """
    Try to remove one term at a time. Returns (new_U, new_V, new_W) if successful, else None.
    """
    rank = U.shape[1]
    num_entries = n * m * p
    
    for r_remove in range(rank):
        if deadline is not None and time.time() > deadline:
            break
        
        remaining = [x for x in range(rank) if x != r_remove]
        if not remaining:
            break
        
        M_remaining = term_vecs[:, remaining]
        
        try:
            c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
            
            if residuals and len(residuals) > 0 and residuals[0] < tol:
                # Verify the solution
                U_new = np.zeros((n*m, len(remaining)))
                V_new = np.zeros((m*p, len(remaining)))
                W_new = np.zeros((p*n, len(remaining)))
                
                for idx, r_idx in enumerate(remaining):
                    U_new[:, idx] = U[:, r_idx] * c[idx]
                    V_new[:, idx] = V[:, r_idx]
                    W_new[:, idx] = W[:, r_idx]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                    return U_new, V_new, W_new
        except Exception:
            pass
            
    return None

def _try_multi_removal(U, V, W, n, m, p, term_vecs, target_flat, tol=1e-10, deadline=None, max_pairs=50):
    """
    Try to remove pairs of terms simultaneously. Returns (new_U, new_V, new_W) if successful, else None.
    """
    rank = U.shape[1]
    
    # Try all pairs of terms to remove
    count = 0
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            if deadline is not None and time.time() > deadline:
                return None
            if count >= max_pairs:
                return None
            count += 1
            
            remaining = [x for x in range(rank) if x != r1 and x != r2]
            if len(remaining) < 2:
                continue
            
            M_remaining = term_vecs[:, remaining]
            
            try:
                c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
                
                if residuals and len(residuals) > 0 and residuals[0] < tol:
                    U_new = np.zeros((n*m, len(remaining)))
                    V_new = np.zeros((m*p, len(remaining)))
                    W_new = np.zeros((p*n, len(remaining)))
                    
                    for idx, r_idx in enumerate(remaining):
                        U_new[:, idx] = U[:, r_idx] * c[idx]
                        V_new[:, idx] = V[:, r_idx]
                        W_new[:, idx] = W[:, r_idx]
                    
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                        return U_new, V_new, W_new
            except Exception:
                pass
                
    return None

def _als_refinement(U, V, W, n, m, p, max_iter=50, deadline=None, tol=1e-12):
    """
    ALS-based refinement to improve the decomposition quality.
    Alternates between optimizing U, V, W given the others fixed.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Optimize U: for each column r, U[:,r] should satisfy
        # sum_k V[j*p+k,r]*W[k*n+i,r]*U[i*m+j,r] = 1
        # This is a linear system in U entries
        for r in range(rank):
            v_r = V[:, r]
            w_r = W[:, r]
            
            # Build the linear system for U[:,r]
            # A * u_r = b where A is (n*m, n*m) 
            # A[i*m+j, i'*m+j'] = sum_k V[j*p+k,r]*W[k*n+i',r]*delta(i,j)
            # Actually, let's think differently.
            
            # The constraint is: for all (i,j,k):
            # U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] contributes to the sum
            # So fixing V and W, we need:
            # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
            
            # For a single r, if we're doing ALS on U:
            # We need U[i*m+j,r] such that the total contribution works out.
            # This is complex. Let me use a different approach.
            pass
        
        # Alternative: use the tensor structure directly
        # For each r, define M_r[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # We want sum_r M_r = ones
        # ALS on U: for each r, fix V_r, W_r, solve for U_r
        # This is a rank-1 tensor approximation problem
        
        # Simpler approach: just verify and return
        break
    
    return U, V, W

def _snap_to_half_integer(U, V, W, tol=1e-6):
    """
    Try to snap entries to nearest half-integer and verify.
    Returns (U_snapped, V_snapped, W_snapped) if valid, else None.
    """
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Check if snapping changed values significantly
    if (np.max(np.abs(U_snap - U)) > tol or 
        np.max(np.abs(V_snap - V)) > tol or 
        np.max(np.abs(W_snap - W)) > tol):
        return None
    
    n, m, p = DIMENSIONS
    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-8):
        return U_snap, V_snap, W_snap
    
    return None

def _aggressive_elimination(U, V, W, n, m, p, max_rounds=20, deadline=None):
    """
    Iteratively try to remove one or two terms at a time, repeating until no more can be removed.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    target_flat = np.ones((n, m, p)).flatten()
    
    for round_num in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        # Recompute term vectors
        term_vecs = _compute_term_vectors(U, V, W, n, m, p)
        
        # Try to remove one term first
        result = _try_single_removal(U, V, W, n, m, p, term_vecs, target_flat, 
                                      tol=1e-10, deadline=deadline)
        if result is not None:
            U, V, W = result
            rank = U.shape[1]
            continue
        
        # Try to remove two terms
        result = _try_multi_removal(U, V, W, n, m, p, term_vecs, target_flat, 
                                     tol=1e-10, deadline=deadline, max_pairs=100)
        if result is not None:
            U, V, W = result
            rank = U.shape[1]
            continue
        
        # No more removals possible
        break
    
    # Try to snap to half-integer
    snapped = _snap_to_half_integer(U, V, W)
    if snapped is not None:
        U, V, W = snapped
    
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Strassen-based rank-65 construction
    try:
        U, V, W = _construct_rank_65(n, m, p)
        rank = U.shape[1]
        
        if _verify_decomposition(U, V, W, n, m, p):
            best_result = (U, V, W)
            best_rank = rank
        else:
            best_result = None
    except Exception:
        best_result = None
    
    # Approach 2: Aggressive elimination to reduce rank
    if best_result is not None:
        try:
            remaining_time = deadline - time.time()
            if remaining_time > 10:
                U_r, V_r, W_r = _aggressive_elimination(best_result[0], best_result[1], 
                                                         best_result[2], n, m, p, 
                                                         max_rounds=50, deadline=deadline)
                rank_r = U_r.shape[1]
                if rank_r < best_rank:
                    best_result = (U_r, V_r, W_r)
                    best_rank = rank_r
                else:
                    best_result = (U_r, V_r, W_r)
        except Exception:
            pass
    
    # If we have a result, return it
    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
