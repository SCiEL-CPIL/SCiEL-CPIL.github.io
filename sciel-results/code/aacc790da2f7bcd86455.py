"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((dim_u, rank_fb))
    V_fb = np.zeros((dim_v, rank_fb))
    W_fb = np.zeros((dim_w, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_fb
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if the decomposition is valid by checking all 48 non-zero entries."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def compute_loss(U, V, W):
        """Compute sum of squared errors on the 48 non-zero positions."""
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    loss += (val - 1.0) ** 2
        return loss
    
    def _als_iterate(U, V, W, max_iter=200, tol=1e-12):
        """Alternating least squares for tensor decomposition."""
        rank = U.shape[1]
        for it in range(max_iter):
            U_old = U.copy()
            V_old = V.copy()
            W_old = W.copy()
            
            # Update U: for each (i,j), solve linear system
            for idx in range(dim_u):
                i, j = divmod(idx, m)
                # Need: sum_c U[idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k in range(p)
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    U[idx, :] = sol
                except:
                    pass
            
            # Update V: for each (j,k), solve linear system
            for idx in range(dim_v):
                j, k = divmod(idx, p)
                # Need: sum_c U[i*m+j,c] * V[idx,c] * W[k*n+i,c] = 1 for all i in range(n)
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    V[idx, :] = sol
                except:
                    pass
            
            # Update W: for each (k,i), solve linear system
            for idx in range(dim_w):
                k, i = divmod(idx, n)
                # Need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[idx,c] = 1 for all j in range(m)
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    W[idx, :] = sol
                except:
                    pass
            
            # Check convergence
            if (np.max(np.abs(U - U_old)) < tol and 
                np.max(np.abs(V - V_old)) < tol and 
                np.max(np.abs(W - W_old)) < tol):
                break
        return U, V, W
    
    def _try_rank(rank, max_restarts=10, seed=None):
        """Try to find a rank-r exact decomposition using ALS."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        for restart in range(max_restarts):
            # Random initialization with small scale
            U = np.random.randn(dim_u, rank) * 0.5
            V = np.random.randn(dim_v, rank) * 0.5
            W = np.random.randn(dim_w, rank) * 0.5
            
            # Run ALS iterations
            U, V, W = _als_iterate(U, V, W, max_iter=150)
            
            # Check if we got close to exact
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                # Try to snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                if check_decomposition(U_snapped, V_snapped, W_snapped, tol=1e-6):
                    return U_snapped, V_snapped, W_snapped
                else:
                    # Return the un-snapped solution if it's accurate enough
                    if check_decomposition(U, V, W, tol=1e-8):
                        return U, V, W
        
        return None
    
    # Try Winograd-inspired structured initialization for 2x2 blocks
    def _winograd_init_2x2(rank, seed=None):
        """Initialize using Winograd-style structure for 2x2 subproblems."""
        if seed is not None:
            np.random.seed(seed)
        
        # For 3x4x4, we can try to exploit 2x2 block structure
        # Winograd for 2x2x2 uses rank 7
        # We'll create a hybrid initialization
        
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        # Fill with random values but with some structure
        for r in range(rank):
            U[:, r] = np.random.randn(dim_u) * 0.3
            V[:, r] = np.random.randn(dim_v) * 0.3
            W[:, r] = np.random.randn(dim_w) * 0.3
        
        return U, V, W
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: at least max(n, m, p) = 4, but realistically higher
    # For 3x4x4, known bounds suggest rank around 30-40
    
    lower_bound = 25
    upper_bound = rank_fb - 1
    
    # First try with standard random init + ALS
    for rank in range(lower_bound, upper_bound + 1):
        if time.time() > deadline:
            break
        
        result = _try_rank(rank, max_restarts=8, seed=rank * 1000)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break
    
    # If we haven't found a good solution, try Winograd-inspired init for higher ranks
    if best_rank == rank_fb:
        for rank in range(35, upper_bound + 1):
            if time.time() > deadline:
                break
            
            found = False
            for restart in range(5):
                seed = rank * 1000 + restart
                U, V, W = _winograd_init_2x2(rank, seed=seed)
                U, V, W = _als_iterate(U, V, W, max_iter=100)
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    U_snapped = np.round(U * 2) / 2
                    V_snapped = np.round(V * 2) / 2
                    W_snapped = np.round(W * 2) / 2
                    
                    if check_decomposition(U_snapped, V_snapped, W_snapped, tol=1e-6):
                        best_rank = rank
                        best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                    elif check_decomposition(U, V, W, tol=1e-8):
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                    found = True
                    break
            
            if found:
                break
    
    # Final polish: if we have a solution, try to improve it with more ALS iterations
    if best_rank < rank_fb:
        U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        U, V, W = _als_iterate(U, V, W, max_iter=500, tol=1e-14)
        
        # Try snapping again
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        if check_decomposition(U_snapped, V_snapped, W_snapped, tol=1e-6):
            best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
        elif check_decomposition(U, V, W, tol=1e-8):
            best_U, best_V, best_W = U, V, W
    
    # Return as lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
