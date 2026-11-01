import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def _build_strassen_35():
    """Return the known rank-35 Strassen decomposition."""
    n, m, p = 2, 2, 10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(n * m):
                U[idx][col] = a_vec[idx]
            V[0 * p + 2 * k][col] = b_vec[0]
            V[0 * p + 2 * k + 1][col] = b_vec[1]
            V[1 * p + 2 * k][col] = b_vec[2]
            V[1 * p + 2 * k + 1][col] = b_vec[3]
            W[2 * k * n + 0][col] = c_vec[0]
            W[2 * k * n + 1][col] = c_vec[1]
            W[(2 * k + 1) * n + 0][col] = c_vec[2]
            W[(2 * k + 1) * n + 1][col] = c_vec[3]
            col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=2, p=10, tol=1e-8):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    T_recon = np.zeros((nm, mp, pn))
    for k in range(rank):
        uk = U_arr[:, k]
        vk = V_arr[:, k]
        wk = W_arr[:, k]
        T_recon += np.einsum('a,b,c->abc', uk, vk, wk)
    
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                T_target[a, b, c] = 1.0
    
    diff = np.abs(T_recon - T_target)
    return np.max(diff) < tol

def _try_reduce_rank(U, V, W, n=2, m=2, p=10, max_target=30, deadline=None):
    """
    Try to reduce the rank of the decomposition by finding linear dependencies.
    
    Strategy: The tensor equation is:
        sum_k U[:,k] * V[:,k] * W[:,k] = T_target
    
    If we have a rank-r decomposition, we can try to find a rank-(r-1) decomposition
    by looking for a vector x such that we can absorb one term.
    
    A more practical approach: try all pairs of columns to see if one can be expressed
    as a linear combination of others in a way that preserves the tensor equation.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    nm, r = U_arr.shape
    mp = V_arr.shape[0]
    pn = W_arr.shape[0]
    
    cur_rank = r
    best_rank = cur_rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Build target tensor for verification
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                T_target[a, b, c] = 1.0
    
    def verify_arrays(U_a, V_a, W_a):
        T_rec = np.zeros((nm, mp, pn))
        for k in range(U_a.shape[1]):
            T_rec += np.einsum('a,b,c->abc', U_a[:,k], V_a[:,k], W_a[:,k])
        return np.max(np.abs(T_rec - T_target)) < 1e-8
    
    # Try to reduce rank by finding exact linear dependencies
    # For each column k, try to express it as a linear combination of others
    # such that the tensor equation still holds.
    
    # The tensor equation for column k being replaced by sum_j alpha_j * column_j:
    # U[:,k]*V[:,k]*W[:,k] = sum_j alpha_j * U[:,j]*V[:,j]*W[:,j]
    
    # This is a nonlinear constraint. Instead, let's try a simpler approach:
    # Look for columns that are linearly dependent in a structured way.
    
    # Alternative: use the fact that the problem has block structure.
    # The 5 blocks (k=0..4) are independent 2x2x2 problems.
    # Each block uses 7 terms. Total = 35.
    # Can we share terms between blocks?
    
    # Let's try: for each pair of blocks, see if we can merge their Strassen terms.
    # Actually, let's just try brute-force rank reduction by checking
    # if any column can be eliminated.
    
    for target_rank in range(cur_rank - 1, max(target_rank - 1, 25), -1):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Try to find a set of columns to eliminate
        # Use a greedy approach: try each column, check if it can be removed
        U_try = best_U.copy()
        V_try = best_V.copy()
        W_try = best_W.copy()
        r = U_try.shape[1]
        
        for elim_count in range(cur_rank - target_rank):
            if r <= target_rank:
                break
            if deadline is not None and time.time() > deadline - 5:
                break
            
            found = False
            for col in range(r):
                # Try to eliminate column col
                # Check if removing it still gives a valid decomposition
                # by seeing if the remaining columns can span the same tensor
                mask = [i for i in range(r) if i != col]
                U_sub = U_try[:, mask]
                V_sub = V_try[:, mask]
                W_sub = W_try[:, mask]
                
                # The contribution of column col is:
                T_col = np.einsum('a,b,c->abc', U_try[:,col], V_try[:,col], W_try[:,col])
                
                # Can the remaining columns reproduce T_col?
                # This is: find alpha such that sum_j alpha_j * U_sub[:,j]*V_sub[:,j]*W_sub[:,j] = T_col
                # This is a linear system in alpha if we flatten the tensor.
                
                T_remaining = np.zeros((nm, mp, pn))
                for j in range(len(mask)):
                    T_remaining += np.einsum('a,b,c->abc', U_sub[:,j], V_sub[:,j], W_sub[:,j])
                
                # We need: T_remaining + T_col = T_target
                # So: T_col must be in the span of the remaining tensor terms
                # But that's not quite right. We need the sum to equal T_target.
                
                # Actually, the full sum should equal T_target:
                T_full = T_remaining + T_col
                if np.max(np.abs(T_full - T_target)) > 1e-8:
                    continue
                
                # Now check if we can find alpha such that:
                # sum_j alpha_j * U_sub[:,j]*V_sub[:,j]*W_sub[:,j] = T_col
                # Flatten: this is a linear system A * alpha = b
                # where A has columns = vec(U_sub[:,j]*V_sub[:,j]*W_sub[:,j])
                
                A = np.zeros((nm*mp*pn, len(mask)))
                for j in range(len(mask)):
                    T_j = np.einsum('a,b,c->abc', U_sub[:,j], V_sub[:,j], W_sub[:,j])
                    A[:, j] = T_j.flatten()
                
                b = T_col.flatten()
                
                # Solve least squares
                try:
                    alpha, residuals, rank_lsq, sv = np.linalg.lstsq(A, b, rcond=None)
                    fit_err = np.linalg.norm(A @ alpha - b)
                    if fit_err < 1e-6:
                        # We can eliminate this column
                        U_try = np.delete(U_try, col, axis=1)
                        V_try = np.delete(V_try, col, axis=1)
                        W_try = np.delete(W_try, col, axis=1)
                        r -= 1
                        found = True
                        break
                except:
                    continue
            
            if not found:
                break
        
        if r < best_rank and verify_arrays(U_try, V_try, W_try):
            best_rank = r
            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
            if best_rank <= target_rank:
                break
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin from 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with the known rank-35 solution
    result = _build_strassen_35()
    best_rank = 35
    
    # Try to reduce rank
    U, V, W = result["U"], result["V"], result["W"]
    
    new_U, new_V, new_W, new_rank = _try_reduce_rank(U, V, W, n, m, p, max_target=30, deadline=deadline)
    
    if new_rank < best_rank:
        best_rank = new_rank
        result = {"U": new_U, "V": new_V, "W": new_W}
    
    return result
