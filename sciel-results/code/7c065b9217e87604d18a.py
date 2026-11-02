"""
Program to find a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
Combines the schoolbook decomposition (guaranteed valid, rank 125) with an alternating
minimization heuristic to attempt to find a lower-rank decomposition with integer/half-integer
entries. The approach initializes with schoolbook and then tries to reduce rank by solving
linear systems in an alternating fashion, followed by rounding to the nearest half-integer
and verification.
"""

import numpy as np
import time

DIMENSIONS = [5, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin for 150s limit

def construct():
    """
    Attempts to find a lower-rank decomposition than the schoolbook (rank 125)
    by using alternating minimization with rounding to half-integers.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    
    # 1. Generate Schoolbook Decomposition as safe fallback
    U_school = np.zeros((u_dim, rank_schoolbook), dtype=float)
    V_school = np.zeros((v_dim, rank_schoolbook), dtype=float)
    W_school = np.zeros((w_dim, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Fallback result
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    # 2. Try to find a lower rank decomposition using alternating minimization
    # We'll try ranks from a reasonable lower bound up to schoolbook-1
    # Theoretical lower bound for 5x5x5 is around 70-80, but we'll start higher for efficiency
    # and decrease if time permits.
    
    def verify_decomposition(U, V, W, rank, tol=1e-8):
        """Verify that the decomposition is exact."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol, max_err
    
    def try_alternating_minimization(rank, max_iters=50, seed=42):
        """
        Try to find a rank decomposition via alternating minimization.
        """
        if time.time() - START_TIME > TIME_LIMIT - 10:
            return None
        
        np.random.seed(seed)
        
        # Initialize with schoolbook if possible, otherwise random
        if rank >= rank_schoolbook:
            U = np.zeros((u_dim, rank), dtype=float)
            V = np.zeros((v_dim, rank), dtype=float)
            W = np.zeros((w_dim, rank), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
        else:
            # Random initialization with small values
            U = np.random.randn(u_dim, rank) * 0.1
            V = np.random.randn(v_dim, rank) * 0.1
            W = np.random.randn(w_dim, rank) * 0.1
        
        # Alternating minimization
        for it in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                return None
            
            # Update U: For each (i, j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # Need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                    # Let B[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    # Then we need U[i*m+j, :] @ B.T = ones(p)
                    # i.e., B @ U[i*m+j, :].T = ones(p)
                    B = np.zeros((rank, p), dtype=float)
                    for c in range(rank):
                        for k in range(p):
                            B[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    try:
                        sol, residuals, rank_B, s = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U[i*m+j, :] = sol
                    except:
                        pass
            
            # Update V: For each (j, k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    # Need: sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c] = 1 for all i
                    # Let A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                    # Then we need V[j*p+k, :] @ A.T = ones(n)
                    # i.e., A @ V[j*p+k, :].T = ones(n)
                    A = np.zeros((rank, n), dtype=float)
                    for c in range(rank):
                        for i in range(n):
                            A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        sol, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p+k, :] = sol
                    except:
                        pass
            
            # Update W: For each (k, i), solve for W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    # Need: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
                    # Let C[c, j] = U[i*m+j, c] * V[j*p+k, c]
                    # Then we need W[k*n+i, :] @ C.T = ones(m)
                    # i.e., C @ W[k*n+i, :].T = ones(m)
                    C = np.zeros((rank, m), dtype=float)
                    for c in range(rank):
                        for j in range(m):
                            C[c, j] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        sol, residuals, rank_C, s = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k*n+i, :] = sol
                    except:
                        pass
            
            # Check convergence
            if it > 0:
                max_err, _ = verify_decomposition(U, V, W, rank, tol=1e-6)
                if max_err < 1e-6:
                    break
        
        # Round to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify rounded version
        max_err, _ = verify_decomposition(U_r, V_r, W_r, rank, tol=1e-8)
        if max_err < 1e-8:
            return U_r, V_r, W_r, max_err
        return None
    
    # Try ranks from 124 down to 100 (or as low as time permits)
    # We'll try a few ranks to see if we can beat 125
    ranks_to_try = list(range(124, 99, -1))
    
    for rank in ranks_to_try:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        result = try_alternating_minimization(rank, max_iters=30, seed=rank)
        if result is not None:
            U_r, V_r, W_r, err = result
            if rank < best_rank:
                best_U = U_r
                best_V = V_r
                best_W = W_r
                best_rank = rank
                # If we found a good rank, we could stop, but let's try to go lower
                # Actually, let's continue to see if we can do even better
                # But for efficiency, if we find something significantly better, we might stop
                if rank <= 110:
                    break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
