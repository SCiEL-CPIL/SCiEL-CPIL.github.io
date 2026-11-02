"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    Uses Winograd-style structured initialization + gradient-based optimization.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 112
    
    def verify_np(U_np, V_np, W_np, tol=1e-9):
        """Verify tensor equation: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def compute_error_fast(Ur, Vr, Wr):
        """Fast error computation using vectorized operations"""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ur[i*m+j], Vr[j*p+k] * Wr[k*n+i])
                    err += (s - 1.0) ** 2
        return err
    
    def compute_gradients(Ur, Vr, Wr):
        """Compute gradients of error w.r.t. U, V, W"""
        rank = Ur.shape[1]
        gU = np.zeros_like(Ur)
        gV = np.zeros_like(Vr)
        gW = np.zeros_like(Wr)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ur[i*m+j], Vr[j*p+k] * Wr[k*n+i])
                    ds = 2.0 * (s - 1.0)
                    # d(err)/d(U[i*m+j, r]) = ds * Vr[j*p+k, r] * Wr[k*n+i, r]
                    gU[i*m+j] += ds * Vr[j*p+k] * Wr[k*n+i]
                    # d(err)/d(V[j*p+k, r]) = ds * Ur[i*m+j, r] * Wr[k*n+i, r]
                    gV[j*p+k] += ds * Ur[i*m+j] * Wr[k*n+i]
                    # d(err)/d(W[k*n+i, r]) = ds * Ur[i*m+j, r] * Vr[j*p+k, r]
                    gW[k*n+i] += ds * Ur[i*m+j] * Vr[j*p+k]
        
        return gU, gV, gW
    
    def gradient_descent(Ur, Vr, Wr, lr=0.01, max_iter=200, tol=1e-10):
        """Gradient descent with adaptive learning rate"""
        current_err = compute_error_fast(Ur, Vr, Wr)
        best_err = current_err
        best_U, best_V, best_W = Ur.copy(), Vr.copy(), Wr.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            if current_err < tol:
                break
            
            gU, gV, gW = compute_gradients(Ur, Vr, Wr)
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
            if grad_norm < 1e-15:
                break
            
            # Step size with line search
            alpha = lr
            for _ in range(5):
                new_U = Ur - alpha * gU
                new_V = Vr - alpha * gV
                new_W = Wr - alpha * gW
                new_err = compute_error_fast(new_U, new_V, new_W)
                if new_err < current_err:
                    Ur, Vr, Wr = new_U, new_V, new_W
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U, best_V, best_W = Ur.copy(), Vr.copy(), Wr.copy()
                    break
                else:
                    alpha *= 0.5
            
            if current_err < 1e-12:
                break
        
        return best_U, best_V, best_W, best_err
    
    # Schoolbook solution as fallback
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    random.seed(42)
    np.random.seed(42)
    
    # Try Winograd-style structured initialization for various ranks
    # Winograd algorithm exploits structure to reduce rank
    # For n x m x p, we can use polynomial multiplication techniques
    
    # Start with reasonable rank estimates
    # Known bounds: 4x4x4 has rank ~23, so 4x4x7 might be around 30-40
    target_ranks = [25, 28, 30, 32, 35, 38, 40, 45, 50]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        if target_rank >= best_rank:
            continue
        
        found = False
        num_restarts = 15
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            # Winograd-style initialization: use structured patterns
            # Based on polynomial multiplication: (a0 + a1*x) * (b0 + b1*x) = c0 + c1*x + c2*x^2
            # We can group indices to exploit this structure
            
            # Initialize with a mix of structured and random components
            U_r = np.random.randn(n * m, target_rank) * 0.3
            V_r = np.random.randn(m * p, target_rank) * 0.3
            W_r = np.random.randn(p * n, target_rank) * 0.3
            
            # Add structured components based on Winograd patterns
            # For each "group" of operations, use shared factors
            group_size = 2  # Winograd with 2x2 blocks
            for r in range(target_rank):
                # Assign structured values to some entries
                if r % group_size == 0:
                    # U: pattern based on i, j
                    for i in range(n):
                        for j in range(m):
                            U_r[i*m+j, r] = ((-1)**i) * ((-1)**j) * (0.5 + 0.1*np.random.randn())
                elif r % group_size == 1:
                    # V: pattern based on j, k
                    for j in range(m):
                        for k in range(p):
                            V_r[j*p+k, r] = ((-1)**j) * ((-1)**k) * (0.5 + 0.1*np.random.randn())
            
            # Run gradient descent
            U_opt, V_opt, W_opt, final_err = gradient_descent(U_r, V_r, W_r, lr=0.02, max_iter=300, tol=1e-10)
            
            if final_err < 1e-8:
                # Verify with tight tolerance
                if verify_np(U_opt, V_opt, W_opt, tol=1e-7):
                    # Polish with tighter tolerance
                    U_polish, V_polish, W_polish, polish_err = gradient_descent(
                        U_opt, V_opt, W_opt, lr=0.005, max_iter=100, tol=1e-12
                    )
                    if verify_np(U_polish, V_polish, W_polish, tol=1e-9):
                        best_U, best_V, best_W = U_polish.copy(), V_polish.copy(), W_polish.copy()
                        best_rank = target_rank
                        found = True
                        break
                    elif verify_np(U_opt, V_opt, W_opt, tol=1e-9):
                        best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                        best_rank = target_rank
                        found = True
                        break
                elif verify_np(U_opt, V_opt, W_opt, tol=1e-9):
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue
    
    # If we found a better solution, try to reduce rank further
    if best_rank < schoolbook_rank:
        for target_rank in range(best_rank - 1, max(best_rank - 5, 15), -1):
            if time.time() > deadline - 10:
                break
            
            found_lower = False
            for restart in range(10):
                if time.time() > deadline - 5:
                    break
                
                U_r = np.random.randn(n * m, target_rank) * 0.3
                V_r = np.random.randn(m * p, target_rank) * 0.3
                W_r = np.random.randn(p * n, target_rank) * 0.3
                
                U_opt, V_opt, W_opt, final_err = gradient_descent(
                    U_r, V_r, W_r, lr=0.02, max_iter=200, tol=1e-10
                )
                
                if final_err < 1e-8 and verify_np(U_opt, V_opt, W_opt, tol=1e-9):
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                    best_rank = target_rank
                    found_lower = True
                    break
            
            if not found_lower:
                break
    
    # Convert to list format
    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
