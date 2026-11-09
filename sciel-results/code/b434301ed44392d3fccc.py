import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    target_rank = 16
    
    def compute_max_error(U, V, W, r):
        """Compute maximum error in tensor reconstruction."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :]) 
                    # Wait, the decomposition is sum_rr U[i,j,rr] V[j,k,rr] W[k,i,rr]
                    # So val = sum_rr U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    # This is not a simple dot product of two vectors.
                    # It is the inner product of (U_row * V_row) with W_row?
                    # Yes: sum_rr (U[i*m+j, rr] * V[j*p+k, rr]) * W[k*n+i, rr]
                    # Let A = U[i*m+j, :] * V[j*p+k, :]
                    # val = np.dot(A, W[k*n+i, :])
                    A = U[i*m+j, :] * V[j*p+k, :]
                    val = np.dot(A, W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def try_rank_als(r, seed=42, max_iters=500):
        """Try to find a rank-r decomposition using vectorized ALS."""
        rng = np.random.RandomState(seed)
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Initialize
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Update U
            # For each j, solve for U[:, j]
            for j in range(m):
                # M[i, :] = V[j, :] * W[i, :]
                # V[j, :] is V[j*p : (j+1)*p, :]? No.
                # V index is j*p + k. So V[j, :] in my notation is V[j*p + k, :] for fixed k?
                # No, in the update for U, we fix V and W.
                # Equation: sum_rr U[i, j, rr] V[j, k, rr] W[k, i, rr] = 1
                # We solve for U[i, j, :] for all i, k? No.
                # For fixed j, we have equations for all i, k.
                # Let's group by j.
                # For a fixed j, we want to find U[i, j, :] for all i.
                # The equations are: for all i, k:
                # sum_rr U[i, j, rr] V[j, k, rr] W[k, i, rr] = 1
                # This couples U[i, j, :] for different i? No, U[i, j, :] only appears in equations with that i.
                # So for fixed j and i, we have p equations (one for each k) for r unknowns U[i, j, :].
                # Let's re-verify the grouping.
                
                # Let's update U[i, j, :] for all i, j.
                # For fixed i, j:
                # Equations: for all k:
                # sum_rr U[i, j, rr] V[j, k, rr] W[k, i, rr] = 1
                # Let v_k = V[j*p + k, :]
                # Let w_k = W[k*n + i, :]
                # Let u = U[i*m + j, :]
                # sum_rr u_rr * v_k_rr * w_k_rr = 1
                # This is a linear system for u.
                # Matrix M of shape (p, r).
                # M[k, :] = v_k * w_k
                # Solve M u = 1_p.
                
                for i in range(n):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V
            # For fixed j, k:
            # Equations: for all i:
            # sum_rr U[i, j, rr] V[j, k, rr] W[k, i, rr] = 1
            # Let u_i = U[i*m + j, :]
            # Let w_i = W[k*n + i, :]
            # Let v = V[j*p + k, :]
            # sum_rr u_i_rr * v_rr * w_i_rr = 1
            # Matrix M of shape (n, r).
            # M[i, :] = u_i * w_i
            # Solve M v = 1_n.
            
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update W
            # For fixed k, i:
            # Equations: for all j:
            # sum_rr U[i, j, rr] V[j, k, rr] W[k, i, rr] = 1
            # Let u_j = U[i*m + j, :]
            # Let v_j = V[j*p + k, :]
            # Let w = W[k*n + i, :]
            # sum_rr u_j_rr * v_j_rr * w_rr = 1
            # Matrix M of shape (m, r).
            # M[j, :] = u_j * v_j
            # Solve M w = 1_m.
            
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Check convergence
            curr_err = compute_max_error(U, V, W, r)
            if curr_err < best_err:
                best_err = curr_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if curr_err < 1e-12:
                break

        if best_U is not None and best_err < 1e-10:
            # Round to half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            max_err_round = compute_max_error(U_rounded, V_rounded, W_rounded, r)
            
            if max_err_round < 1e-10:
                return U_rounded, V_rounded, W_rounded, True
            else:
                # If rounding fails, return the floating point solution?
                # The task requires integer/half-integer entries.
                # So we must return rounded if it works.
                # If it doesn't work, this attempt failed.
                return None, None, None, False
        
        return None, None, None, False

    # Try rank 16
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        U, V, W, success = try_rank_als(target_rank, seed=seed, max_iters=300)
        if success:
            U_list = [[float(U[i][j]) for j in range(target_rank)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(target_rank)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(target_rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
