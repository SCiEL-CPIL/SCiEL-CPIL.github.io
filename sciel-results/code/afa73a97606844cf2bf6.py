import numpy as np
import time

def construct():
    """
    Rank-reduced decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Kronecker-Strassen construction then systematic column elimination.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    U_base, V_base, W_base = _construct_rank98(n, m, p)
    
    # Verify the base decomposition
    if not _verify_decomposition(U_base, V_base, W_base, n, m, p):
        U_base, V_base, W_base = _construct_naive(n, m, p)
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    best_result = (U_base, V_base, W_base)
    best_rank = U_base.shape[1]
    
    # Try column elimination to reduce rank
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 10:
        reduced = _column_elimination_aggressive(U_base, V_base, W_base, n, m, p, deadline - start_time)
        if reduced is not None:
            U_r, V_r, W_r = reduced
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                if U_r.shape[1] < best_rank:
                    best_result = (U_r, V_r, W_r)
                    best_rank = U_r.shape[1]
    
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}

def _construct_rank98(n, m, p):
    """Rank-98 decomposition using two independent 4x4 Strassen instances."""
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
    
    # Build 4x4x4 rank-49 decomposition
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

def _column_elimination_aggressive(U, V, W, n, m, p, time_budget):
    """
    Aggressive column elimination.
    """
    start = time.time()
    rank = U.shape[1]
    num_eqs = n * m * p
    
    # Build tensor matrix T
    T = np.zeros((num_eqs, rank), dtype=float)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for t in range(rank):
                    T[eq_idx, t] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                eq_idx += 1
    
    target = np.ones(num_eqs, dtype=float)
    
    if not np.allclose(T.sum(axis=1), target, atol=1e-9):
        return None
        
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_T = T.copy()
    current_rank = rank
    
    for _ in range(current_rank):
        if time.time() - start > time_budget:
            break
            
        removed = False
        for r in range(current_rank):
            # Check if column r is in span of others
            if current_rank <= 1:
                break
            remaining = [s for s in range(current_rank) if s != r]
            T_rem = current_T[:, remaining]
            T_target = current_T[:, r]
            
            try:
                alpha, residuals, rank_A, sv = np.linalg.lstsq(T_rem, T_target, rcond=1e-10)
            except:
                continue
                
            if residuals.size > 0:
                res_val = residuals[0]
            else:
                res_val = np.linalg.norm(T_rem @ alpha - T_target)
                
            if res_val > 1e-8:
                continue
                
            # Check if sum of remaining equals target
            # This is a sufficient condition for simple removal
            # But we can also just remove if the relation holds and update?
            # No, simple removal only works if T_r is "redundant" in the sum.
            # Actually, if T_r = sum alpha_s T_s, then sum T_s' = sum T_s - T_r + sum alpha_s T_s?
            # No.
            # If we remove r, the new sum is sum_{s!=r} T_s.
            # We need sum_{s!=r} T_s = target.
            # This implies T_r = 0.
            # So simple removal only works for zero columns.
            
            # To actually reduce rank, we need to find a linear dependence among the columns
            # that allows us to express one column's contribution as a combination of others,
            # AND modify the other columns to compensate.
            # This is complex.
            
            # Alternative: Check if the rank of the tensor is less than current_rank.
            # We can compute the rank of the matrix T (flattened tensor).
            # But T is already the flattened tensor.
            # The rank of the decomposition is the number of columns.
            # If the columns of T are linearly dependent, we might be able to reduce rank.
            # However, the U, V, W factors must also be consistent.
            
            # Let's try a different approach:
            # If T_r is in the span of other T_s, we can set U_r = 0, V_r = 0, W_r = 0?
            # No, that changes the sum.
            
            # For now, only remove if T_r is zero.
            if np.allclose(T_target, 0, atol=1e-9):
                keep = [s for s in range(current_rank) if s != r]
                current_U = current_U[:, keep]
                current_V = current_V[:, keep]
                current_W = current_W[:, keep]
                current_T = current_T[:, keep]
                current_rank -= 1
                removed = True
                break
                
        if not removed:
            break
            
    if current_rank < rank:
        return current_U, current_V, current_W
    return None

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
