"""Program to find a low-rank exact decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = 3, 3, 5
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if the decomposition is exact: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
                    if max_err > tol:
                        return max_err
        return max_err
    
    def round_to_half_integers(U, V, W):
        """Round entries to nearest half-integers and verify."""
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        return U_r, V_r, W_r
    
    def verify_half_integer(U, V, W, tol=1e-8):
        """Check if entries are half-integers and decomposition is exact."""
        # Check half-integer constraint
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if not np.allclose(U2, np.round(U2), atol=1e-6):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-6):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-6):
            return False
        # Check decomposition
        return check_decomposition(U, V, W, tol) < tol
    
    def polish(U, V, W, max_iters=50, tol=1e-10):
        """High-precision polish: alternate least squares with tight tolerance."""
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            # Solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, U.shape[1]))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = x
            U = U_new
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, V.shape[1]))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = x
            V = V_new
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, W.shape[1]))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = x
            W = W_new
            
            err = check_decomposition(U, V, W, tol=float('inf'))
            if err < tol:
                break
        
        return U, V, W, check_decomposition(U, V, W, tol=float('inf'))
    
    def alternating_minimization(rank, max_iters=200, seed=42):
        """ALS for tensor decomposition with random initialization."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n*m, rank) * 0.1
        V = rng.randn(m*p, rank) * 0.1
        W = rng.randn(p*n, rank) * 0.1
        
        # Normalize columns to avoid scale issues
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            total = norm_u * norm_v * norm_w
            if total > 1e-12:
                U[:, r] /= norm_u
                V[:, r] /= norm_v
                W[:, r] /= norm_w
                # Scale one factor to absorb the total
                W[:, r] *= total
        
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            # Solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = x
            U = U_new
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = x
            V = V_new
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = x
            W = W_new
            
            err = check_decomposition(U, V, W, tol=float('inf'))
            if err < 1e-6:
                # Try polish
                U, V, W, final_err = polish(U, V, W, max_iters=30, tol=1e-10)
                if final_err < 1e-8:
                    # Try rounding to half-integers
                    U_r, V_r, W_r = round_to_half_integers(U, V, W)
                    if verify_half_integer(U_r, V_r, W_r, tol=1e-6):
                        return U_r, V_r, W_r
                    # Return unrounded if close enough
                    if final_err < 1e-8:
                        return U, V, W
                break
            if abs(err - prev_err) < 1e-12:
                break
            prev_err = err
        
        # Final attempt with polish
        U, V, W, final_err = polish(U, V, W, max_iters=20, tol=1e-10)
        if final_err < 1e-6:
            U_r, V_r, W_r = round_to_half_integers(U, V, W)
            if verify_half_integer(U_r, V_r, W_r, tol=1e-6):
                return U_r, V_r, W_r
            if final_err < 1e-6:
                return U, V, W
        return None, None, None
    
    def schoolbook_decomposition():
        """Return rank-45 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Try to find low-rank decompositions
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 45
    
    # Try ranks from 37 up to 44
    for rank in range(37, 45):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        # Try multiple random seeds
        for seed in range(30):
            if time.time() > deadline - 10.0:
                break
            
            U_try, V_try, W_try = alternating_minimization(rank, max_iters=150, seed=seed)
            if U_try is not None:
                # Verify
                if check_decomposition(U_try, V_try, W_try, tol=1e-6) < 1e-6:
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    # If we found something, return it
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback: Schoolbook
    U, V, W = schoolbook_decomposition()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
