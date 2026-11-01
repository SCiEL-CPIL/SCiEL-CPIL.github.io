"""Matrix multiplication 2,2,6 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 2, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x6 matrix.
    
    The multiplication tensor T has T[i,j,k] = 1 if the (i,j) entry of A
    multiplied by the (j,k) entry of B contributes to the (i,k) entry of C.
    
    We seek rank-r factors U (nm x r), V (mp x r), W (pn x r) such that:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For n=2, m=2, p=6: nm=4, mp=12, pn=12, tensor size 4*12*12 = 576 entries.
    The schoolbook rank is n*m*p = 24. We search for lower ranks.
    
    Strategy: Try ranks from a lower bound upward. For each rank, use random
    initialization with alternating least squares to try to find an exact solution.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the multiplication tensor T_full
    # T_full[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T_full[a, b, c] = 1 if indices are consistent
    T_full = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
    
    # Precompute flattened views for faster ALS
    T_flat_U = T_full.reshape(nm, mp * pn)       # for updating U
    T_flat_V = T_full.transpose(1, 0, 2).reshape(mp, nm * pn)  # for updating V
    T_flat_W = T_full.transpose(2, 0, 1).reshape(pn, nm * mp)  # for updating W
    
    # Schoolbook fallback (rank = n*m*p = 24)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        max_err = 0.0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U[a], V[b] * W[c])
                    err = abs(val - T_full[a, b, c])
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    def try_rank(rank, n_trials=50, seed=None):
        """Try to find a rank-r decomposition via random init + ALS."""
        if seed is not None:
            np.random.seed(seed)
        
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Scale to avoid numerical issues
            U /= np.sqrt(rank)
            V /= np.sqrt(rank)
            W /= np.sqrt(rank)
            
            # Alternating least squares
            converged = False
            for iteration in range(300):
                if time.time() > deadline - 5:
                    break
                
                # Update U: T_flat_U[a, :] = U[a, :] @ M_U.T
                # where M_U[b*p+n+c, r] = V[b, r] * W[c, r]
                M_U = np.zeros((mp * pn, rank))
                for b in range(mp):
                    for c in range(pn):
                        M_U[b * pn + c, :] = V[b, :] * W[c, :]
                
                try:
                    A_U = M_U.T @ M_U
                    B_U = M_U.T @ T_flat_U.T
                    U_new = np.linalg.solve(A_U, B_U).T
                except np.linalg.LinAlgError:
                    U_new = U + 0.01 * np.random.randn(nm, rank)
                
                # Update V: T_flat_V[b, :] = V[b, :] @ M_V.T
                # where M_V[a*p+n+c, r] = U_new[a, r] * W[c, r]
                M_V = np.zeros((nm * pn, rank))
                for a in range(nm):
                    for c in range(pn):
                        M_V[a * pn + c, :] = U_new[a, :] * W[c, :]
                
                try:
                    A_V = M_V.T @ M_V
                    B_V = M_V.T @ T_flat_V.T
                    V_new = np.linalg.solve(A_V, B_V).T
                except np.linalg.LinAlgError:
                    V_new = V + 0.01 * np.random.randn(mp, rank)
                
                # Update W: T_flat_W[c, :] = W[c, :] @ M_W.T
                # where M_W[a*m*p+b, r] = U_new[a, r] * V_new[b, r]
                M_W = np.zeros((nm * mp, rank))
                for a in range(nm):
                    for b in range(mp):
                        M_W[a * mp + b, :] = U_new[a, :] * V_new[b, :]
                
                try:
                    A_W = M_W.T @ M_W
                    B_W = M_W.T @ T_flat_W.T
                    W_new = np.linalg.solve(A_W, B_W).T
                except np.linalg.LinAlgError:
                    W_new = W + 0.01 * np.random.randn(pn, rank)
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                error = 0.0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            error += (val - T_full[a, b, c]) ** 2
                
                if error < 1e-10:
                    converged = True
                    break
            
            if converged:
                if verify(U, V, W):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
        return None
    
    start_time = time.time()
    deadline = start_time + 110  # 110 second limit (safety margin)
    
    # Try ranks from lower to higher
    start_rank = max(nm, mp, pn)  # = 12
    end_rank = n * m * p - 1  # = 23 (skip schoolbook, we have fallback)
    
    best_result = None
    for rank in range(start_rank, end_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, n_trials=30, seed=rank * 1000)
        if result is not None:
            best_result = result
            break
    
    # If no lower rank found, return schoolbook
    if best_result is None:
        best_result = schoolbook()
    
    return best_result
