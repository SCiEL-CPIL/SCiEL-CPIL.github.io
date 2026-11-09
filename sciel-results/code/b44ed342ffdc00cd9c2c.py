import numpy as np
import time
from itertools import product
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 130.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
        
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)
    
    def compute_tensor(U, V, W):
        """Compute the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
        n, m, p = DIMENSIONS
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
        return T
    
    def try_rational_rounding(U, V, W, max_denom=16):
        """Try to round to rational numbers with small denominators and verify."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        U_rounded = np.zeros_like(U)
        V_rounded = np.zeros_like(V)
        W_rounded = np.zeros_like(W)
        
        for denom in [1, 2, 4, 8, 16]:
            scale = denom
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            
            if verify_decomposition(U_r, V_r, W_r, tol=1e-10):
                return U_r, V_r, W_r, True
        
        return None, None, None, False
    
    def tensor_unfolding_search(target_rank, max_iters=200, tol=1e-6):
        """Use iterative refinement based on tensor unfolding."""
        n, m, p = DIMENSIONS
        
        # Initialize with random values
        np.random.seed(int(time.time() * 1000) % 10000)
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        prev_error = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Compute current tensor
            T = compute_tensor(U, V, W)
            error = np.sum((T - 1.0) ** 2)
            
            if abs(error - prev_error) < 1e-12:
                break
            prev_error = error
            
            if error < tol:
                # Try to round to rational
                U_r, V_r, W_r, success = try_rational_rounding(U, V, W)
                if success:
                    return U_r, V_r, W_r, True
                return U, V, W, verify_decomposition(U, V, W, tol=1e-8)
            
            # Update W: W[k*n+i, r] should satisfy sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1
            # For fixed U, V, we can solve for W row by row
            W_new = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    # W[k*n+i, :] needs to satisfy: for all j, sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1
                    # This is a system: sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1 for j=0..m-1
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A @ W[k*n+i, :] = ones(m)
                    try:
                        W_row, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = W_row
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            
            # Update U: similar logic
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = W[k * n + i, :] * V[j * p + k, :]
                    try:
                        U_row, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = U_row
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            
            # Update V: similar logic
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V_row, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = V_row
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            
            # Damped update
            alpha = 0.5
            U = alpha * U_new + (1 - alpha) * U
            V = alpha * V_new + (1 - alpha) * V
            W = alpha * W_new + (1 - alpha) * W
        
        # Final check
        T = compute_tensor(U, V, W)
        error = np.sum((T - 1.0) ** 2)
        if error < 1e-6:
            U_r, V_r, W_r, success = try_rational_rounding(U, V, W)
            if success:
                return U_r, V_r, W_r, True
            return U, V, W, verify_decomposition(U, V, W, tol=1e-8)
        
        return None, None, None, False
    
    # Try structured search from rank 83 down
    for target_rank in range(rank_school - 1, 40, -1):
        if time.time() > deadline - 30:
            break
        
        # Try tensor unfolding search
        U_sol, V_sol, W_sol, success = tensor_unfolding_search(target_rank, max_iters=100)
        
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
