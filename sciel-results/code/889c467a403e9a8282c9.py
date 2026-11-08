import numpy as np
import time

DIMENSIONS = [3, 4, 5]
TIME_LIMIT = 145  # safety margin from 150s

def _verify_fast(U, V, W):
    """Fast verification using matrix multiplication.
    
    The tensor decomposition says:
    T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    
    This is equivalent to: for each pair (u,w), the vector over v is:
    (U[:,r] * W[w,r]) outer product summed over r
    
    Equivalently: M[u,v] = sum_r U[u,r]*V[v,r] = (U @ V.T)[u,v]
    And we need M[u,v] * W[w,r] summed over r to give the tensor.
    
    Better: T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    For fixed (u,w): T[u,:,w] = sum_r U[u,r]*W[w,r] * V[:,r] = (U[u,:] * W[w,:]) @ V.T
    
    So for each (u,w), check if (U[u,:]*W[w,:]) @ V.T equals the expected row.
    
    Even faster: Build M = U @ V.T (shape n*m x m*p)
    Then T[u,v,w] = sum_r M[u,v]_r ... no that's not right.
    
    Actually: T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    Let P = U @ V.T, so P[u,v] = sum_r U[u,r]*V[v,r]
    That's not directly useful because we have three factors.
    
    Correct approach: For each w, T[:,:,w] = sum_r (U[:,r] * V[:,r]) * W[w,r]
    = sum_r W[w,r] * (U[:,r] outer V[:,r])
    
    Alternatively: T[u,v,w] = sum_r (U[u,r]*W[w,r]) * V[v,r]
    For fixed u: T[u,v,w] = sum_r (U[u,r]*W[w,r]) * V[v,r]
    Let Q[u,w] vector = U[u,:] * W[w,:] (element-wise, shape rank)
    Then T[u,:,w] = Q[u,w] @ V.T (shape m*p)
    
    So: T = sum over u,w of Q[u,w] outer (V.T) ... this is getting complex.
    
    Simplest fast check: 
    For each (u, w) pair, compute row = U[u,:] * W[w,:] (element-wise product, length rank)
    Then T[u,:,w] should equal row @ V.T
    
    We need to check that T[u,v,w] = 1 if (u,v,w) corresponds to valid matmul indices, else 0.
    
    Let's just do: for each valid (i,l,k), check T[i*m+l, l*p+k, k*n+i] = 1
    And for invalid positions, check T = 0.
    
    But we can vectorize the valid checks:
    valid_u = [i*m+l for i in range(n) for l in range(m) for k in range(p)]
    valid_v = [l*p+k for i in range(n) for l in range(m) for k in range(p)]
    valid_w = [k*n+i for i in range(n) for l in range(m) for k in range(p)]
    
    Then T[valid_u, valid_v, valid_w] should all be 1.
    
    For the invalid check, we need T to be 0 everywhere else.
    """
    n, m, p = 3, 4, 5
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid positions: T[i*m+l, l*p+k, k*n+i] = 1
    # T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    
    # Vectorized: for each (i,l,k), compute the sum
    # u = i*m+l, v = l*p+k, w = k*n+i
    # T = sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r]
    
    # Build index arrays
    ii = np.repeat(np.arange(n), m*p)
    ll = np.tile(np.repeat(np.arange(m), p), n)
    kk = np.tile(np.arange(p), n*m)
    
    u_idx = ii * m + ll
    v_idx = ll * p + kk
    w_idx = kk * n + ii
    
    # T_valid = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
    # = np.einsum('ur,ur,ur->u', U[u_idx], V[v_idx], W[w_idx])
    T_valid = np.einsum('ir,jr,kr->i', U[u_idx], V[v_idx], W[w_idx])
    
    if np.any(np.abs(T_valid - 1.0) > 1e-9):
        return False
    
    # Check invalid positions are zero
    # Total positions: n*m * m*p * p*n = 12*20*15 = 3600
    # Valid positions: n*m*p = 60
    # Invalid: 3540
    
    # Build full tensor and check
    # T_full[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    # Use einsum: 'ur,vr,wr->uvw'
    T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    # Create mask
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    mask[u_idx, v_idx, w_idx] = True
    
    invalid = ~mask
    if np.any(np.abs(T_full[invalid]) > 1e-9):
        return False
    
    return True

def _verify_matrix(U, V, W):
    """Alternative verification using matrix products.
    
    T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    
    For fixed w: T[:,:,w] = sum_r W[w,r] * (U[:,r] outer V[:,r])
    = sum_r W[w,r] * np.outer(U[:,r], V[:,r])
    
    This can be written as: T[:,:,w] = (U * W[w,:]) @ V.T
    where U * W[w,:] means U[:,r] * W[w,r] for each r.
    
    So: T[:,:,w] = (U * W[w][None,:]) @ V.T
    """
    n, m, p = 3, 4, 5
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    # For each w in range(p*n):
    # T[:,:,w] = (U * W[w][None,:]) @ V.T
    # shape: (n*m, rank) * (rank,) -> (n*m, rank), then @ (rank, m*p) -> (n*m, m*p)
    
    # We need T[i*m+l, l*p+k, k*n+i] = 1 and T = 0 elsewhere
    
    # Check valid positions
    ii = np.repeat(np.arange(n), m*p)
    ll = np.repeat(np.repeat(np.arange(m), p), n)
    kk = np.tile(np.arange(p), n*m)
    
    u_idx = ii * m + ll
    v_idx = ll * p + kk
    w_idx = kk * n + ii
    
    # For each valid (u,v,w) triple, check T[u,v,w] = 1
    # T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    T_valid = np.einsum('ir,jr,kr->i', U[u_idx], V[v_idx], W[w_idx])
    
    if np.any(np.abs(T_valid - 1.0) > 1e-9):
        return False
    
    # For invalid positions, T should be 0
    # Use the matrix formulation: for each w, T[:,:,w] = (U * W[w][None,:]) @ V.T
    # Check that all entries except the valid ones are 0
    
    for w in range(p * n):
        T_w = (U * W[w][None, :]) @ V.T  # shape (n*m, m*p)
        # Zero out valid positions for this w
        valid_v_for_w = np.where(w_idx == w)[0]
        if len(valid_v_for_w) > 0:
            for idx in valid_v_for_w:
                T_w[u_idx[idx], v_idx[idx]] = 0.0
        if np.any(np.abs(T_w) > 1e-9):
            return False
    
    return True

def _seed_solution():
    """Return the seed 56-rank solution as U,V,W numpy arrays."""
    n, m, p = 3, 4, 5
    U_parts = []
    V_parts = []
    W_parts = []
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W
    
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_blk[i, r]
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_blk[i, r]
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_blk[i, r]
    
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], *strassen_2x2x2())
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], *strassen_2x2x2())
    add_block([0,1,4,5], [4,9], [12,13], *schoolbook_2x2x1())
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], *strassen_2x2x2())
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], *strassen_2x2x2())
    add_block([2,3,6,7], [14,19], [12,13], *schoolbook_2x2x1())
    add_block([8,9], [0,1,5,6], [2,5], *schoolbook_1x2x2())
    add_block([8,9], [2,3,7,8], [8,11], *schoolbook_1x2x2())
    add_block([8,9], [4,9], [14], *schoolbook_1x2x1())
    add_block([10,11], [10,11,15,16], [2,5], *schoolbook_1x2x2())
    add_block([10,11], [12,13,17,18], [8,11], *schoolbook_1x2x2())
    add_block([10,11], [14,19], [14], *schoolbook_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W

def _try_remove_single(U, V, W, r):
    """Try removing column r. Returns (U,V,W) if valid, else None."""
    U_test = np.delete(U, r, axis=1)
    V_test = np.delete(V, r, axis=1)
    W_test = np.delete(W, r, axis=1)
    if _verify_fast(U_test, V_test, W_test):
        return U_test, V_test, W_test
    return None

def _try_remove_multiple(U, V, W, cols_to_remove):
    """Try removing multiple columns at once."""
    mask = np.ones(U.shape[1], dtype=bool)
    for r in cols_to_remove:
        mask[r] = False
    U_test = U[:, mask]
    V_test = V[:, mask]
    W_test = W[:, mask]
    if _verify_fast(U_test, V_test, W_test):
        return U_test, V_test, W_test
    return None

def _find_redundant_columns(U, V, W):
    """Use linear algebra to find potentially redundant columns.
    
    A column r is redundant if there exist coefficients a_s (s != r) such that:
    U[:,r] = sum_s a_s * U[:,s]
    V[:,r] = sum_s a_s * V[:,s]  
    W[:,r] = sum_s a_s * W[:,s]
    with the same coefficients a_s.
    
    This is a system of linear equations. Let's set up the problem.
    """
    n, m, p = 3, 4, 5
    rank = U.shape[1]
    
    # For each column r, check if it's in the span of others with consistent coefficients
    # This is expensive, so we'll use a heuristic: check if U[:,r] is in span of other U columns
    # and if so, find the coefficients, then check if they work for V and W.
    
    redundant = set()
    for r in range(rank):
        # Check if U[:,r] is in span of U[:,s] for s != r
        other_cols = [s for s in range(rank) if s != r]
        U_other = U[:, other_cols]  # shape (n*m, rank-1)
        U_r = U[:, r]  # shape (n*m,)
        
        # Solve U_other @ x = U_r
        try:
            x, residuals, rank_u, sv = np.linalg.lstsq(U_other, U_r, rcond=1e-10)
            if residuals.size > 0 and residuals[0] > 1e-8:
                continue  # U[:,r] not in span
            
            # Check if same x works for V and W
            V_other = V[:, other_cols]
            V_r = V[:, r]
            V_pred = V_other @ x
            if np.any(np.abs(V_pred - V_r) > 1e-8):
                continue
            
            W_other = W[:, other_cols]
            W_r = W[:, r]
            W_pred = W_other @ x
            if np.any(np.abs(W_pred - W_r) > 1e-8):
                continue
            
            # Column r is redundant
            redundant.add(r)
        except:
            continue
    
    return redundant

def _try_remove_pairs(U, V, W):
    """Try removing pairs of columns."""
    rank = U.shape[1]
    for i in range(rank):
        for j in range(i+1, rank):
            result = _try_remove_multiple(U, V, W, [i, j])
            if result is not None:
                return result
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start from seed
    U, V, W = _seed_solution()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = U.shape[1]
    
    # Verify seed
    if not _verify_fast(best[0], best[1], best[2]):
        raise ValueError("Seed solution failed verification")
    
    print(f"Initial rank: {best_rank}")
    
    # Iteratively try to remove columns
    for iteration in range(200):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
        
        current_U, current_V, current_W = best
        
        # First try single column removal
        found = False
        for r in range(current_U.shape[1]):
            result = _try_remove_single(current_U, current_V, current_W, r)
            if result is not None:
                best = result
                best_rank = best[0].shape[1]
                found = True
                print(f"Iter {iteration}: removed col {r}, rank now {best_rank}")
                break
        
        if found:
            continue
        
        # Try pair removal
        result = _try_remove_pairs(current_U, current_V, current_W)
        if result is not None:
            best = result
            best_rank = best[0].shape[1]
            print(f"Iter {iteration}: removed pair, rank now {best_rank}")
            continue
        
        # Try finding redundant columns via linear algebra
        redundant = _find_redundant_columns(current_U, current_V, current_W)
        if redundant:
            result = _try_remove_multiple(current_U, current_V, current_W, list(redundant))
            if result is not None:
                best = result
                best_rank = best[0].shape[1]
                print(f"Iter {iteration}: removed redundant {redundant}, rank now {best_rank}")
                continue
        
        # No more progress
        break
    
    elapsed = time.time() - start_time
    print(f"Final rank: {best_rank}, time: {elapsed:.1f}s")
    
    U_out, V_out, W_out = best
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
