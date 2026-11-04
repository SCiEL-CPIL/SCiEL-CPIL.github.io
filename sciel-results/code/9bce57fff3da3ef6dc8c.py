"""Rewritten program for matrix multiplication tensor rank search (3x4 by 4x4)."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
# n=3, m=4, p=4
# U: (n*m, rank) = (12, rank)
# V: (m*p, rank) = (16, rank)
# W: (p*n, rank) = (12, rank)
# Tensor T[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,4)
# Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    # Reconstruct T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # For each r: outer product of U[:,r] reshaped (n,m), V[:,r] reshaped (m,p), W[:,r] reshaped (p,n)
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    # Target is all ones
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _verify_exact_rational(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0])
    T = [[0]*p for _ in range(n*m)]
    # Actually let's build a 3D array
    T3 = np.zeros((n, m, p), dtype=object)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i*m + j][r]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[j*p + k][r]
                    if v_val == 0:
                        continue
                    w_val = W[k*n + i][r]
                    if w_val == 0:
                        continue
                    T3[i, j, k] += u_val * v_val * w_val
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T3[i, j, k] != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_r_random(n, m, p, r, num_trials=10, seed=42, max_iter=50):
    """
    Try to find a rank-r decomposition via random initialization + ALS.
    Returns (U, V, W) as lists of lists of floats if found, else None.
    """
    rng = np.random.RandomState(seed)
    target = _build_target_tensor(n, m, p)
    
    best_err = float('inf')
    best_result = None
    
    # Pre-compute indices for faster access if needed, but direct slicing is fine
    
    for trial in range(num_trials):
        # Random initialization
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        # ALS iterations
        for it in range(max_iter):
            # Update U
            new_U = U.copy()
            for i in range(n):
                for j in range(m):
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    V_slice = V[j*p : (j+1)*p, :] # (p, r)
                    W_indices = np.arange(p) * n + i
                    W_slice = W[W_indices, :] # (p, r)
                    A = V_slice * W_slice # (p, r)
                    b = target[i, j, :] # (p,)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        new_U[i * m + j, :] = x
                    except:
                        pass
            
            # Update V
            new_V = V.copy()
            for j in range(m):
                for k in range(p):
                    # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    U_slice = U[:, j] if False else None # Need to gather U[i*m+j] for all i
                    # U indices: i*m + j for i in range(n)
                    U_indices = np.arange(n) * m + j
                    U_slice = U[U_indices, :] # (n, r)
                    W_indices = np.arange(n) * n + i # Wait, W index is k*n + i. i is fixed in outer loop? No, i is variable here.
                    # For fixed j, k:
                    # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # i varies from 0 to n-1
                    W_indices = k * n + np.arange(n) # (n,)
                    W_slice = W[W_indices, :] # (n, r)
                    A = U_slice * W_slice # (n, r)
                    b = target[:, j, k] # (n,)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        new_V[j * p + k, :] = x
                    except:
                        pass
            
            # Update W
            new_W = W.copy()
            for k in range(p):
                for i in range(n):
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # j varies from 0 to m-1
                    U_indices = i * m + np.arange(m) # (m,)
                    U_slice = U[U_indices, :] # (m, r)
                    V_indices = np.arange(m) * p + k # (m,)
                    V_slice = V[V_indices, :] # (m, r)
                    A = U_slice * V_slice # (m, r)
                    b = target[i, :, k] # (m,)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        new_W[k * n + i, :] = x
                    except:
                        pass
            
            U, V, W = new_U, new_V, new_W
            
            # Check error occasionally to save time
            if it % 10 == 0 or it == max_iter - 1:
                T_recon = np.zeros((n, m, p))
                for r_idx in range(r):
                    u = U[:, r_idx].reshape(n, m)
                    v = V[:, r_idx].reshape(m, p)
                    w = W[:, r_idx].reshape(p, n)
                    T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
                err = np.linalg.norm(T_recon - target)
                if err < best_err:
                    best_err = err
                    best_result = (U.copy(), V.copy(), W.copy())
                if err < 1e-8:
                    break
        
        if best_err < 1e-8:
            break
    
    if best_result is None:
        return None
    
    U_arr, V_arr, W_arr = best_result
    # Final verification
    T_recon = np.zeros((n, m, p))
    for r_idx in range(r):
        u = U_arr[:, r_idx].reshape(n, m)
        v = V_arr[:, r_idx].reshape(m, p)
        w = W_arr[:, r_idx].reshape(p, n)
        T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
    err = np.linalg.norm(T_recon - target)
    if err < 1e-6:
        U_list = [[float(x) for x in row] for row in U_arr]
        V_list = [[float(x) for x in row] for row in V_arr]
        W_list = [[float(x) for x in row] for row in W_arr]
        return {"U": U_list, "V": V_list, "W": W_list}
    return None

def construct():
    deadline = time.time() + 110  # 10s safety margin
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # Let's try ranks from 16 upward
    
    for r in range(16, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try random ALS for this rank
        # Reduced trials and iterations to ensure speed
        result = _try_rank_r_random(n, m, p, r, num_trials=10, seed=42 + r, max_iter=50)
        if result is not None:
            # Verify it's valid
            U, V, W = result["U"], result["V"], result["W"]
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-5):
                best = result
                break  # Found the lowest rank, stop
    
    return best
