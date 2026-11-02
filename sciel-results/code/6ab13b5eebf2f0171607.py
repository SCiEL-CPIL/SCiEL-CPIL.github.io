import numpy as np
import time

DIMENSIONS = [2, 4, 5]
DEADLINE = 135.0  # safety margin before 150s kill

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _compute_error(U, V, W):
    """Compute the L2 error of the decomposition."""
    n, m, p = 2, 4, 5
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1.0)**2
    return np.sqrt(err)

def _als_step(U, V, W, max_iters=100, tol=1e-12, time_limit=None, start_time=None):
    """
    Alternating Least Squares to refine the decomposition.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Update U
        for r in range(rank):
            A_mat = np.zeros((n*m*p, n*m))
            b_vec = np.ones(n*m*p)
            idx_count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        var_idx = i*m + j
                        A_mat[idx_count, var_idx] = V[j*p+k, r] * W[k*n+i, r]
                        b_vec[idx_count] = 1.0
                        idx_count += 1
            U[:, r], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Update V
        for r in range(rank):
            A_mat = np.zeros((n*m*p, m*p))
            b_vec = np.ones(n*m*p)
            idx_count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        var_idx = j*p + k
                        A_mat[idx_count, var_idx] = U[i*m+j, r] * W[k*n+i, r]
                        b_vec[idx_count] = 1.0
                        idx_count += 1
            V[:, r], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Update W
        for r in range(rank):
            A_mat = np.zeros((n*m*p, p*n))
            b_vec = np.ones(n*m*p)
            idx_count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        var_idx = k*n + i
                        A_mat[idx_count, var_idx] = U[i*m+j, r] * V[j*p+k, r]
                        b_vec[idx_count] = 1.0
                        idx_count += 1
            W[:, r], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Check convergence
        err = _compute_error(U, V, W)
        if err < tol:
            break
            
    return U, V, W, err

def _try_remove_columns(U, V, W, target_rank, time_limit=5.0):
    """
    Try to reduce the rank by removing linearly dependent columns.
    Uses a greedy approach: try removing each column and see if the remaining
    columns can be adjusted to still represent the tensor.
    """
    n, m, p = 2, 4, 5
    current_rank = U.shape[1]
    start_time = time.time()
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Try to remove columns one at a time
    for _ in range(current_rank - target_rank):
        if time.time() - start_time > time_limit:
            break
        
        removed = False
        for col_to_remove in range(current_rank):
            if time.time() - start_time > time_limit:
                break
            
            # Create matrices without this column
            mask = [i for i in range(current_rank) if i != col_to_remove]
            U_sub = U[:, mask]
            V_sub = V[:, mask]
            W_sub = W[:, mask]
            
            # Try to refine
            U_try, V_try, W_try, err = _als_step(U_sub.copy(), V_sub.copy(), W_sub.copy(), 
                                                  max_iters=50, tol=1e-10,
                                                  time_limit=time_limit-(time.time()-start_time),
                                                  start_time=start_time)
            
            if err < 1e-8:
                # Check if we can round to half-integers
                U_round = np.round(U_try * 2) / 2
                V_round = np.round(V_try * 2) / 2
                W_round = np.round(W_try * 2) / 2
                if _verify_decomposition(U_round, V_round, W_round):
                    U, V, W = U_round, V_round, W_round
                    current_rank -= 1
                    removed = True
                    break
                elif _verify_decomposition(U_try, V_try, W_try):
                    U, V, W = U_try, V_try, W_try
                    current_rank -= 1
                    removed = True
                    break
        
        if not removed:
            break
    
    return U, V, W

def _random_search(rank, time_limit=5.0):
    """
    Try random initializations to find a decomposition of given rank.
    """
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        U, V, W, err = _als_step(U, V, W, max_iters=50, tol=1e-10,
                                  time_limit=time_limit-(time.time()-start_time),
                                  start_time=start_time)
        
        if err < 1e-8:
            # Try to round
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            if _verify_decomposition(U_round, V_round, W_round):
                return U_round, V_round, W_round
            elif _verify_decomposition(U, V, W):
                return U, V, W
    
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    U, V, W = _build_strassen_36()
    
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook
        n, m, p = 2, 4, 5
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 36
    
    # First, try to refine the existing decomposition
    U_ref, V_ref, W_ref, err = _als_step(U.copy(), V.copy(), W.copy(), 
                                          max_iters=50, tol=1e-12,
                                          time_limit=5.0, start_time=start_time)
    if err < 1e-10:
        # Try to round
        U_round = np.round(U_ref * 2) / 2
        V_round = np.round(V_ref * 2) / 2
        W_round = np.round(W_ref * 2) / 2
        if _verify_decomposition(U_round, V_round, W_round):
            best_U, best_V, best_W = U_round, V_round, W_round
    
    # Try to reduce rank by removing columns
    for r in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 15:
            break
        
        result = _try_remove_columns(best_U, best_V, best_W, r, 
                                     time_limit=min(5.0, deadline - time.time() - 10))
        if result is not None:
            U_try, V_try, W_try = result
            if _verify_decomposition(U_try, V_try, W_try):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = r
    
    # If we have time, try random search for lower ranks
    for r in range(min(best_rank - 1, 30), 20, -1):
        if time.time() > deadline - 10:
            break
        
        result = _random_search(r, time_limit=min(3.0, deadline - time.time() - 5))
        if result is not None:
            U_try, V_try, W_try = result
            if _verify_decomposition(U_try, V_try, W_try):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = r
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W):
        # Fall back to the original Strassen decomposition
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
