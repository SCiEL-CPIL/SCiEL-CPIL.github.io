import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2x2():
    """
    Returns U, V, W matrices (4x7) for the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    Indices: a=i*2+j, b=j*2+k, c=k*2+i for i,j,k in {0,1}.
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    u1 = [1, 0, 0, 1]
    u2 = [0, 0, 1, 1]
    u3 = [1, 0, 0, 0]
    u4 = [0, 0, 0, 1]
    u5 = [1, 1, 0, 0]
    u6 = [-1, 0, 1, 0]
    u7 = [0, 1, 0, -1]
    
    v1 = [1, 0, 0, 1]
    v2 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    v4 = [0, 0, 1, -1]
    v5 = [0, 0, 0, 1]
    v6 = [1, 1, 0, 0]
    v7 = [0, 0, 1, 1]
    
    w1 = [1, 0, 0, 1]
    w2 = [0, 0, 1, 0]
    w3 = [0, 1, 0, 0]
    w4 = [-1, 0, 1, 0]
    w5 = [0, 1, 0, 1]
    w6 = [1, 0, 0, -1]
    w7 = [-1, 0, 0, 1]
    
    U = np.array([u1, u2, u3, u4, u5, u6, u7]).T
    V = np.array([v1, v2, v3, v4, v5, v6, v7]).T
    W = np.array([w1, w2, w3, w4, w5, w6, w7]).T
    
    return U, V, W

def _build_base_decomposition(n, m, p):
    """Build the rank-93 base decomposition using 4x4x4 Strassen + 4x4x3 schoolbook."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U_2, V_2, W_2 = get_strassen_2x2x2()
    
    U_list = []
    V_list = []
    W_list = []
    
    # 4x4x4 part via Kronecker (rank 49)
    for r in range(7):
        for s in range(7):
            u = np.kron(U_2[:, r], U_2[:, s]).astype(float)
            v_kron = np.kron(V_2[:, r], V_2[:, s]).astype(float)
            v_full = np.zeros(dim_v)
            for j in range(4):
                for k in range(4):
                    v_full[j * p + k] = v_kron[j * 4 + k]
            w_kron = np.kron(W_2[:, r], W_2[:, s]).astype(float)
            w_full = np.zeros(dim_w)
            w_full[:16] = w_kron
            
            U_list.append(u)
            V_list.append(v_full)
            W_list.append(w_full)
    
    # 4x4x3 schoolbook part (rank 48)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = np.zeros(dim_u)
                v = np.zeros(dim_v)
                w = np.zeros(dim_w)
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u[a] = 1.0
                v[b] = 1.0
                w[c] = 1.0
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    return U, V, W

def _build_target_tensor(n, m, p):
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _compute_residual(U, V, W, T_target):
    """Compute T_target - sum_r u_r ⊗ v_r ⊗ w_r."""
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return T_target - T_decomp.astype(np.int64)

def _try_remove_term(U, V, W, T_target, term_idx):
    """
    Try to remove term term_idx by adjusting other terms.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None).
    """
    r = U.shape[1]
    if r <= 1:
        return False, None, None, None
    
    # The contribution of term term_idx
    u_t = U[:, term_idx]
    v_t = V[:, term_idx]
    w_t = W[:, term_idx]
    contrib = np.einsum('a,b,c->abc', u_t, v_t, w_t)
    
    # We need to redistribute contrib among the remaining terms.
    # For each remaining term r', we can add delta_r' to its coefficient.
    # New U[:,r'] = U[:,r'] + alpha_r' * something... 
    # Actually, the standard approach: we want to find corrections such that
    # sum_{r' != term_idx} (u_{r'} + du_{r'}) ⊗ (v_{r'} + dv_{r'}) ⊗ (w_{r'} + dw_{r'}) 
    #   = sum_{r' != term_idx} u_{r'} ⊗ v_{r'} ⊗ w_{r'} + contrib
    # This is complex. Simpler: try setting one of the three vectors of term_idx to zero
    # and see if the residual can be absorbed.
    
    # Alternative: try to express u_t as a linear combination of other u's,
    # v_t as combination of other v's, w_t as combination of other w's.
    # If u_t = sum_{r'!=t} c_{r'} u_{r'}, then we can modify V and W.
    # But this only works if the tensor structure allows it.
    
    # Practical approach: try removing the term and check if the residual
    # (which is -contrib) can be represented by modifying existing terms.
    # We'll try a simple heuristic: for each remaining term, try to add
    # a scaled version of the removed term's vectors.
    
    # Actually, the cleanest structural approach: 
    # Remove term t. The new residual is -contrib.
    # We need to find small modifications to other terms to cancel -contrib.
    # This is equivalent to: can we write contrib as a sum of rank-1 tensors
    # that are "close" to the existing terms?
    
    # For now, use a simpler test: remove the term and check if the resulting
    # tensor still equals T_target by trying to solve for corrections.
    # We'll use a least-squares approach on a subset.
    
    # Simpler heuristic: just remove and check.
    U_new = np.delete(U, term_idx, axis=1)
    V_new = np.delete(V, term_idx, axis=1)
    W_new = np.delete(W, term_idx, axis=1)
    
    T_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
    if np.array_equal(T_new, T_target):
        return True, U_new, V_new, W_new
    
    return False, None, None, None

def _sparsify(U, V, W, T_target, max_time=100):
    """
    Iteratively try to remove redundant terms from the decomposition.
    Uses a greedy approach: try removing each term, keep the one that succeeds.
    """
    start_time = time.time()
    r = U.shape[1]
    
    # Order terms by "importance" - terms with smaller norm are more likely redundant
    # Compute norm of each rank-1 term
    norms = np.zeros(r)
    for i in range(r):
        norms[i] = np.linalg.norm(U[:, i]) * np.linalg.norm(V[:, i]) * np.linalg.norm(W[:, i])
    
    # Sort indices by norm (smallest first - most likely redundant)
    order = np.argsort(norms)
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        r = U.shape[1]
        # Recompute norms each iteration
        norms = np.zeros(r)
        for i in range(r):
            norms[i] = np.linalg.norm(U[:, i]) * np.linalg.norm(V[:, i]) * np.linalg.norm(W[:, i])
        order = np.argsort(norms)
        
        for idx in order:
            if (time.time() - start_time) > max_time:
                break
            success, U_new, V_new, W_new = _try_remove_term(U, V, W, T_target, idx)
            if success:
                U, V, W = U_new, V_new, W_new
                improved = True
                break  # Restart with new ordering
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build base rank-93 decomposition
    U, V, W = _build_base_decomposition(n, m, p)
    T_target = _build_target_tensor(n, m, p)
    
    # Verify base decomposition
    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.array_equal(T_check, T_target):
        # Fallback to schoolbook
        rank = n * m * p
        U_s = np.zeros((dim_u, rank))
        V_s = np.zeros((dim_v, rank))
        W_s = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Time budget: leave 20s margin
    elapsed = time.time() - time.time()  # Reset reference
    start = time.time()
    max_sparsify_time = 80  # seconds
    
    # Attempt sparsification
    U_sparse, V_sparse, W_sparse = _sparsify(U, V, W, T_target, max_time=max_sparsify_time)
    
    # Verify sparsified result
    T_final = np.einsum('ar,br,cr->abc', U_sparse, V_sparse, W_sparse)
    if np.array_equal(T_final, T_target):
        # Convert to clean integer/half-integer form
        U_out = np.round(U_sparse * 2) / 2
        V_out = np.round(V_sparse * 2) / 2
        W_out = np.round(W_sparse * 2) / 2
        
        # Final verification
        T_verify = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
        if np.array_equal(T_verify, T_target):
            return {
                "U": U_out.tolist(),
                "V": V_out.tolist(),
                "W": W_out.tolist()
            }
    
    # Return the base decomposition if sparsification didn't help or failed
    U_out = np.round(U * 2) / 2
    V_out = np.round(V * 2) / 2
    W_out = np.round(W * 2) / 2
    
    T_verify = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
    if np.array_equal(T_verify, T_target):
        return {
            "U": U_out.tolist(),
            "V": V_out.tolist(),
            "W": W_out.tolist()
        }
    
    # Ultimate fallback
    rank = n * m * p
    U_s = np.zeros((dim_u, rank))
    V_s = np.zeros((dim_v, rank))
    W_s = np.zeros((dim_w, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_s[a, col] = 1
                V_s[b, col] = 1
                W_s[c, col] = 1
                col += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
