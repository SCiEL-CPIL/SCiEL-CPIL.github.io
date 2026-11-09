"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses Vectorized Alternating Least Squares (ALS) to find low-rank decompositions.
    """
    start_time = time.time()
    deadline = 140  # Keep 10s safety margin before 150s kill
    n, m, p = 3, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 12, 20, 15
    
    # Indices for tensor T[i,j,k] = 1
    # We want to minimize sum_{i,j,k} (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1)^2
    
    def check_solution(U, V, W, rank):
        """Compute max absolute error."""
        # Vectorized check
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :rank], V[v_idx, :rank] * W[w_idx, :rank])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def solve_als(rank, seed, max_iters=1000, time_budget=10.0):
        """
        Solve for U, V, W using ALS.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        reg = 1e-6
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget + start_time:
                break
            
            # Update U: For each (i,j), solve U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :])^T = 1
            # This is a least squares problem: A_u @ u = b_u
            # A_u has shape (p, rank), b_u has shape (p,)
            for i in range(n):
                for j in range(m):
                    u_idx = i*m + j
                    A_u = np.zeros((p, rank))
                    for k in range(p):
                        v_idx = j*p + k
                        w_idx = k*n + i
                        A_u[k, :] = V[v_idx, :] * W[w_idx, :]
                    b_u = np.ones(p)
                    # Solve (A_u^T A_u + reg I) u = A_u^T b_u
                    try:
                        U[u_idx, :] = np.linalg.solve(A_u.T @ A_u + reg * np.eye(rank), A_u.T @ b_u)
                    except np.linalg.LinAlgError:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    v_idx = j*p + k
                    A_v = np.zeros((n, rank))
                    for i in range(n):
                        u_idx = i*m + j
                        w_idx = k*n + i
                        A_v[i, :] = U[u_idx, :] * W[w_idx, :]
                    b_v = np.ones(n)
                    try:
                        V[v_idx, :] = np.linalg.solve(A_v.T @ A_v + reg * np.eye(rank), A_v.T @ b_v)
                    except np.linalg.LinAlgError:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    w_idx = k*n + i
                    A_w = np.zeros((m, rank))
                    for j in range(m):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        A_w[j, :] = U[u_idx, :] * V[v_idx, :]
                    b_w = np.ones(m)
                    try:
                        W[w_idx, :] = np.linalg.solve(A_w.T @ A_w + reg * np.eye(rank), A_w.T @ b_w)
                    except np.linalg.LinAlgError:
                        pass
            
            # Check error
            curr_err = check_solution(U, V, W, rank)
            if curr_err < best_err:
                best_err = curr_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
                
        return best_err, best_U, best_V, best_W

    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from low to high
    # Theoretical lower bound is max(n*m, m*p, p*n) ? No, it's related to matrix multiplication complexity.
    # For 3x4x5, Strassen-like algorithms might help, but simple rank minimization is hard.
    # Let's try a range.
    ranks_to_try = [20, 25, 30, 35, 40, 45, 50, 55, 60]
    
    for rank in ranks_to_try:
        if time.time() - start_time > deadline - 5:
            break
        
        # Try multiple seeds for each rank
        found = False
        for seed in [42, 123, 999, 2024]:
            if time.time() - start_time > deadline - 5:
                break
            
            # Allocate time per attempt
            time_left = deadline - (time.time() - start_time) - 2
            if time_left < 1:
                break
            
            err, U_c, V_c, W_c = solve_als(rank, seed, max_iters=200, time_budget=time_left/4.0)
            
            if err < 1e-10:
                best_rank = rank
                best_U, best_V, best_W = U_c, V_c, W_c
                found = True
                break
        
        if found:
            break
            
    # Final verification
    if best_rank < rank_schoolbook:
        final_err = check_solution(best_U, best_V, best_W, best_rank)
        if final_err > 1e-8:
            # Fallback
            best_rank = rank_schoolbook
            best_U, best_V, best_W = U_sb, V_sb, W_sb

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    U_out = to_list(best_U[:, :best_rank])
    V_out = to_list(best_V[:, :best_rank])
    W_out = to_list(best_W[:, :best_rank])
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
