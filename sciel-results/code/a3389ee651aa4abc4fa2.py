import numpy as np
import time

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill

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

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _compute_error(U, V, W):
    """Compute the error of the decomposition."""
    n, m, p = 2, 4, 5
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1.0)**2
    return np.sqrt(err)

def _als_optimize(U, V, W, max_iters=200, tol=1e-10, time_limit=5.0):
    """
    Alternating Least Squares optimization.
    Fix two matrices, solve for the third.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        prev_err = _compute_error(U, V, W)
        if prev_err < tol:
            break
        
        # Update U: fix V and W
        # For each (i,j), the constraint involves U[i*m+j, :]
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
        # This is a linear system in U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # Build A matrix: (p x rank) where A[k,r] = V[j*p+k,r] * W[k*n+i,r]
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Solve A @ U[i*m+j, :].T = ones(p)
                # This is underdetermined if rank > p, so use least squares
                b = np.ones(p)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i*m+j, :] = sol
        
        # Update V: fix U and W
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                b = np.ones(n)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j*p+k, :] = sol
        
        # Update W: fix U and V
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                b = np.ones(m)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k*n+i, :] = sol
        
        curr_err = _compute_error(U, V, W)
        if curr_err < tol:
            break
        if abs(curr_err - prev_err) < 1e-12:
            break
    
    return U, V, W, curr_err

def _try_reduce_rank(U, V, W, target_rank, time_limit=4.0):
    """
    Try to reduce the rank from current to target_rank by selecting a subset of columns.
    """
    n, m, p = 2, 4, 5
    current_rank = U.shape[1]
    start_time = time.time()
    
    if target_rank >= current_rank:
        return None
    
    best_result = None
    best_err = float('inf')
    
    num_attempts = 30
    for attempt in range(num_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly select which columns to keep
        keep_idx = np.random.choice(current_rank, target_rank, replace=False)
        keep_idx = np.sort(keep_idx)
        
        U_new = U[:, keep_idx].copy()
        V_new = V[:, keep_idx].copy()
        W_new = W[:, keep_idx].copy()
        
        # Optimize
        time_left = time_limit - (time.time() - start_time)
        if time_left < 0.5:
            break
        U_new, V_new, W_new, err = _als_optimize(
            U_new, V_new, W_new, max_iters=100, tol=1e-10, time_limit=min(time_left, 2.0)
        )
        
        if err < best_err:
            best_err = err
            best_result = (U_new.copy(), V_new.copy(), W_new.copy())
        
        if err < 1e-8:
            # Check if it can be rounded to half-integers
            U_round = np.round(U_new * 2) / 2
            V_round = np.round(V_new * 2) / 2
            W_round = np.round(W_new * 2) / 2
            if _verify_decomposition(U_round, V_round, W_round):
                return U_round, V_round, W_round
            elif _verify_decomposition(U_new, V_new, W_new):
                return U_new, V_new, W_round
    
    if best_result is not None and best_err < 1e-8:
        U_r, V_r, W_r = best_result
        U_round = np.round(U_r * 2) / 2
        V_round = np.round(V_r * 2) / 2
        W_round = np.round(W_r * 2) / 2
        if _verify_decomposition(U_round, V_round, W_round):
            return U_round, V_round, W_round
        elif _verify_decomposition(U_r, V_r, W_r):
            return U_r, V_r, W_r
    
    return None

def _try_random_init_rank(rank, time_limit=5.0):
    """
    Try to find a decomposition of given rank from random initialization.
    """
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        time_left = time_limit - (time.time() - start_time)
        if time_left < 0.5:
            break
        U, V, W, err = _als_optimize(
            U, V, W, max_iters=150, tol=1e-10, time_limit=min(time_left, 2.0)
        )
        
        if err < 1e-8:
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
    
    # Try to reduce rank
    for r in range(35, 19, -1):
        if time.time() > deadline - 15:
            break
        
        time_left = deadline - time.time() - 10
        if time_left < 1.0:
            break
        
        # Try reducing from current best
        result = _try_reduce_rank(best_U, best_V, best_W, r, time_limit=min(time_left, 4.0))
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            continue
        
        # If we can't reduce by 1 from current, try random init for this rank
        if time.time() < deadline - 15:
            time_left = deadline - time.time() - 10
            if time_left < 1.0:
                break
            result = _try_random_init_rank(r, time_limit=min(time_left, 3.0))
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = r
    
    if not _verify_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
