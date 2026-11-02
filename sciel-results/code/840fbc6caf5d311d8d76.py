"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition for n x m * m x p."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check that the tensor decomposition matches the matrix multiplication tensor exactly."""
    R = U.shape[1]
    Ur = U.reshape(n, m, R)
    Vr = V.reshape(m, p, R)
    Wr = W.reshape(p, n, R)
    
    # Valid terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += Ur[i, j, r] * Vr[j, k, r] * Wr[k, i, r]
                if abs(s - 1.0) > tol:
                    return False
                    
    # Invalid terms
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            if j == j2 and i == i2 and k == k2:
                                continue
                            s = 0.0
                            for r in range(R):
                                s += Ur[i, j, r] * Vr[j2, k, r] * Wr[k, i2, r]
                            if abs(s) > tol:
                                return False
    return True

def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(x * 2.0) / 2.0

def _als_decompose(n, m, p, R, max_iters=100, tol=1e-8, rng=None):
    """
    Attempt to find a rank-R decomposition using Alternating Least Squares.
    Returns (U, V, W) if a solution with low error is found, else None.
    """
    if rng is None:
        rng = np.random.RandomState()
        
    # Precompute tensor indices and target values
    # T[i, j, j2, k, k2, i2] = 1 if i==i2, j==j2, k==k2 else 0
    # We flatten this to a vector of size n*m * m*p * p*n
    # But we only need the linear system matrix A_W, A_U, A_V
    
    # Dimensions
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    dim_T = dim_U * dim_V * dim_W
    
    # Initialize
    U = rng.randn(dim_U, R) * 0.5
    V = rng.randn(dim_V, R) * 0.5
    W = rng.randn(dim_W, R) * 0.5
    
    prev_err = float('inf')
    
    for it in range(max_iters):
        # --- Solve for W ---
        # System: sum_r U[i,j,r] V[j2,k,r] W[k2,i2,r] = T[i,j,j2,k,k2,i2]
        # Variables: W[k2,i2,r]
        # We build A_W of shape (dim_T, dim_W * R)
        # This is expensive to build every time. 
        # Optimization: We can precompute the structure, but U and V change.
        # For small dimensions (9, 12, 12), building 1296 x 12R matrix is fast enough.
        
        A_W = np.zeros((dim_T, dim_W * R))
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i2 in range(n):
                                row_idx = (i * m + j) * dim_V * dim_W + (j2 * p + k) * dim_W + (k2 * n + i2)
                                for r in range(R):
                                    col_idx = (k2 * n + i2) * R + r
                                    A_W[row_idx, col_idx] = U[i * m + j, r] * V[j2 * p + k, r]
        
        # Target vector T
        T_vec = np.zeros(dim_T)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i, j, j, k, k, i] = 1
                    row_idx = (i * m + j) * dim_V * dim_W + (j * p + k) * dim_W + (k * n + i)
                    T_vec[row_idx] = 1.0
        
        # Solve A_W W_flat = T_vec
        W_flat, residuals, rank, sv = np.linalg.lstsq(A_W, T_vec, rcond=None)
        W = W_flat.reshape(dim_W, R)
        
        # --- Solve for U ---
        A_U = np.zeros((dim_T, dim_U * R))
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i2 in range(n):
                                row_idx = (i * m + j) * dim_V * dim_W + (j2 * p + k) * dim_W + (k2 * n + i2)
                                for r in range(R):
                                    col_idx = (i * m + j) * R + r
                                    A_U[row_idx, col_idx] = V[j2 * p + k, r] * W[k2 * n + i2, r]
        
        U_flat, residuals, rank, sv = np.linalg.lstsq(A_U, T_vec, rcond=None)
        U = U_flat.reshape(dim_U, R)
        
        # --- Solve for V ---
        A_V = np.zeros((dim_T, dim_V * R))
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i2 in range(n):
                                row_idx = (i * m + j) * dim_V * dim_W + (j2 * p + k) * dim_W + (k2 * n + i2)
                                for r in range(R):
                                    col_idx = (j2 * p + k) * R + r
                                    A_V[row_idx, col_idx] = U[i * m + j, r] * W[k2 * n + i2, r]
        
        V_flat, residuals, rank, sv = np.linalg.lstsq(A_V, T_vec, rcond=None)
        V = V_flat.reshape(dim_V, R)
        
        # Check error
        # Reconstruct T_approx
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        
        err = 0.0
        # Check valid terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += Ur[i, j, r] * Vr[j, k, r] * Wr[k, i, r]
                    err += (s - 1.0)**2
        # Check invalid terms (sample or all? All is 1296, fast)
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for i2 in range(n):
                            for k2 in range(p):
                                if j == j2 and i == i2 and k == k2:
                                    continue
                                s = 0.0
                                for r in range(R):
                                    s += Ur[i, j, r] * Vr[j2, k, r] * Wr[k, i2, r]
                                err += s**2
                                
        if err < tol:
            return U, V, W
            
        if abs(prev_err - err) < 1e-10:
            break
        prev_err = err
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Schoolbook Decomposition (Rank 36) - always valid
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = n * m * p  # 36
    
    # 2. Try to find lower rank decompositions using ALS
    # We try ranks from 35 down to 20
    # Time budget per rank: ~5 seconds
    for R in range(35, 19, -1):
        if time.time() > deadline:
            break
            
        time_budget = min(5.0, deadline - time.time())
        if time_budget < 1.0:
            break
            
        # Try a few random restarts
        num_restarts = 5
        for attempt in range(num_restarts):
            if time.time() > start_time + time_budget + 10: # rough check
                 break
            # Use different seeds
            rng = np.random.RandomState(42 + R * 100 + attempt)
            max_iters = 50
            result = _als_decompose(n, m, p, R, max_iters=max_iters, tol=1e-8, rng=rng)
            
            if result is not None:
                U, V, W = result
                # Round to half-integers
                U_rounded = _round_to_half(U)
                V_rounded = _round_to_half(V)
                W_rounded = _round_to_half(W)
                
                # Verify exactness
                if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    best_rank = R
                    break # Found a solution for this rank, try lower rank
            
    return best_result
# EVOLVE-BLOCK-END
