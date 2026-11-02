"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 130.0  # Leave 20s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T[i, j, k] = 1 for all valid i, j, k
    T = np.ones((n, m, p), dtype=float)
    
    # Fallback: Schoolbook decomposition (Rank = n * m * p)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_s, V_s, W_s, rank_s = schoolbook()
    best_result = (U_s, V_s, W_s, rank_s)
    
    # Function to verify a decomposition
    def verify(U, V, W, r):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - T[i, j, k])
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Winograd-inspired approach: exploit structure to find lower-rank decompositions
    # For 2x3x10, we can use the factorization structure
    # 2 = 2, 3 = 3, 10 = 2*5
    # This suggests using tensor product structure
    
    # Strategy 1: Direct structured search using Kronecker products
    # If we can write 2x3x10 as a tensor product of smaller problems
    # 2x3x10 ≈ (2x1x2) ⊗ (1x3x5) or similar decompositions
    
    # Let's try a more systematic approach:
    # For rank r, we need to find U (6 x r), V (30 x r), W (20 x r)
    # such that for all i in [0,1], j in [0,2], k in [0,9]:
    # sum_c U[i*3+j, c] * V[j*10+k, c] * W[k*2+i, c] = 1
    
    # Use alternating minimization with structured initialization
    
    # Lower bound: rank >= max(n, m, p) = 10
    # But we can likely do better with structure
    
    # Try ranks from 10 upwards, with more restarts for smaller ranks
    for r in range(10, min(41, n*m*p + 1)):
        if time.time() > deadline - 20:
            break
            
        found = False
        
        # Number of random restarts - more for smaller ranks
        max_restarts = 50 if r < 15 else 20 if r < 20 else 10
        
        for trial in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize with structured random values
            # Use small integer/half-integer values for better conditioning
            scale = 1.0 if r >= 15 else 0.5
            
            U = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(nm, r)) * scale
            V = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(mp, r)) * scale
            
            # Solve for W given U and V
            W = np.zeros((pn, r))
            success = True
            
            for k in range(p):
                for i in range(n):
                    # Construct A_ki of shape (m, r)
                    # A_ki[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    A_ki = np.zeros((m, r))
                    for j in range(m):
                        A_ki[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    
                    # Solve A_ki @ w = ones(m)
                    # Check if system is consistent
                    try:
                        # First check rank
                        rank_A = np.linalg.matrix_rank(A_ki, tol=1e-10)
                        if rank_A < m:
                            # Underdetermined or singular - use lstsq but check residual
                            w, residuals, rank_lsq, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                            if len(residuals) > 0 and residuals[0] > 1e-8:
                                success = False
                                break
                            W[k*n + i, :] = w
                        else:
                            # Full column rank in m - should have unique solution if consistent
                            w, residuals, rank_lsq, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                            if len(residuals) > 0 and residuals[0] > 1e-8:
                                success = False
                                break
                            W[k*n + i, :] = w
                    except np.linalg.LinAlgError:
                        success = False
                        break
                if not success:
                    break
            
            if not success:
                continue
                
            # Verify
            err = verify(U, V, W, r)
            if err < 1e-9:
                # Polish with exact solve
                U_final, V_final, W_final, rank_final = _polish(U, V, W, r, n, m, p)
                if U_final is not None:
                    best_result = (U_final, V_final, W_final, rank_final)
                    found = True
                    break
        
        if found:
            break
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _polish(U, V, W, r, n, m, p):
    """Try to polish the solution to exact precision"""
    nm, mp, pn = n*m, m*p, p*n
    
    # Verify current solution
    def verify(U, V, W, r):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    err = verify(U, V, W, r)
    if err < 1e-10:
        return U, V, W, r
    
    # Try alternating minimization to polish
    for _ in range(10):
        # Fix V, W and solve for U
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                # U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :]) summed over k = 1
                # For each k, this is: sum_k U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :])
                # = U[i*m+j, :] * sum_k (V[j*p+k, :] * W[k*n+i, :])
                # Let B = sum_k (V[j*p+k, :] * W[k*n+i, :])
                # Then U[i*m+j, :] * B = 1
                B = np.zeros(r)
                for k in range(p):
                    B += V[j*p + k, :] * W[k*n + i, :]
                if np.linalg.norm(B) < 1e-15:
                    continue
                # Solve U[i*m+j, :] * B = 1
                # This is a linear equation in r unknowns - underdetermined
                # Use lstsq
                w, res, rank, sv = np.linalg.lstsq(B.reshape(1, r), np.array([1.0]), rcond=None)
                U_new[i*m + j, :] = w[0]
        U = U_new
        
        # Fix U, W and solve for V
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                # sum_i U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :] = 1
                # V[j*p+k, :] * sum_i (U[i*m+j, :] * W[k*n+i, :]) = 1
                B = np.zeros(r)
                for i in range(n):
                    B += U[i*m + j, :] * W[k*n + i, :]
                if np.linalg.norm(B) < 1e-15:
                    continue
                w, res, rank, sv = np.linalg.lstsq(B.reshape(1, r), np.array([1.0]), rcond=None)
                V_new[j*p + k, :] = w[0]
        V = V_new
        
        # Fix U, V and solve for W
        W_new = W.copy()
        for k in range(p):
            for i in range(n):
                # sum_j U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :] = 1
                # W[k*n+i, :] * sum_j (U[i*m+j, :] * V[j*p+k, :]) = 1
                B = np.zeros(r)
                for j in range(m):
                    B += U[i*m + j, :] * V[j*p + k, :]
                if np.linalg.norm(B) < 1e-15:
                    continue
                w, res, rank, sv = np.linalg.lstsq(B.reshape(1, r), np.array([1.0]), rcond=None)
                W_new[k*n + i, :] = w[0]
        W = W_new
        
        err = verify(U, V, W, r)
        if err < 1e-12:
            break
    
    if verify(U, V, W, r) < 1e-10:
        return U, V, W, r
    return None, None, None, r

# EVOLVE-BLOCK-END
