"""Optimized tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    where the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    equals 1 for all valid (i,j,k) and 0 otherwise.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Tensor dimensions
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    deadline = time.time() + 120.0
    
    # Build the target tensor in flattened form
    # T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # The tensor has shape (dim_u, dim_v, dim_w) where:
    #   u_idx = i*m + j, v_idx = j*p + k, w_idx = k*n + i
    # T[u,v,w] = 1 if there exist i,j,k such that u=i*m+j, v=j*p+k, w=k*n+i
    # and the indices are consistent (same i,j,k)
    
    def build_target():
        """Build the multiplication tensor as a 3D array."""
        T = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[u_idx, v_idx, w_idx] = 1.0
        return T
    
    T = build_target()
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that U, V, W exactly decompose the multiplication tensor."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    max_err = max(max_err, abs(s - 1.0))
        # Also check that all other (u,v,w) combinations give 0
        # This is expensive, so we'll check a sample
        for u in range(dim_u):
            for v in range(dim_v):
                for w in range(dim_w):
                    # Check if this (u,v,w) corresponds to a valid (i,j,k)
                    i = u // m
                    j = u % m
                    j2 = v // p
                    k = v % p
                    k2 = w // n
                    i2 = w % n
                    if i == j2 and j == k2 and i2 == j:
                        # This is a valid tensor entry, should be 1
                        s = np.dot(U[u, :], V[v, :] * W[w, :])
                        max_err = max(max_err, abs(s - 1.0))
                    else:
                        # Should be 0
                        s = np.dot(U[u, :], V[v, :] * W[w, :])
                        max_err = max(max_err, abs(s))
        return max_err < tol
    
    def snap_to_rational(matrix, max_denom=8):
        """Snap matrix entries to nearest rational with small denominator."""
        snapped = np.zeros_like(matrix)
        for idx in np.ndindex(matrix.shape):
            val = matrix[idx]
            best = 0.0
            best_err = abs(val)
            for d in range(1, max_denom + 1):
                num = round(val * d)
                candidate = num / d
                err = abs(val - candidate)
                if err < best_err:
                    best_err = err
                    best = candidate
            snapped[idx] = best
        return snapped
    
    def try_rank(rank, T, max_iters=5000, lr=0.01, seed=42):
        """Try to find a rank-factorization at the given rank using gradient descent."""
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(dim_u, rank) * 0.1
        V = np.random.randn(dim_v, rank) * 0.1
        W = np.random.randn(dim_w, rank) * 0.1
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Precompute valid indices
        valid_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_indices.append((i * m + j, j * p + k, k * n + i))
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            # Compute gradient
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for (u, v, w) in valid_indices:
                # T[u,v,w] should be 1
                # d/du: V[v,:] * W[w,:]
                for r in range(rank):
                    diff = U[u, r] * V[v, r] * W[w, r] - 1.0
                    grad_U[u, r] += diff * V[v, r] * W[w, r]
                    grad_V[v, r] += diff * U[u, r] * W[w, r]
                    grad_W[w, r] += diff * U[u, r] * V[v, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Compute error
            err = 0.0
            for (u, v, w) in valid_indices:
                s = np.dot(U[u, :], V[v, :] * W[w, :])
                err += (s - 1.0) ** 2
            err = np.sqrt(err / len(valid_indices))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
        
        return best_U, best_V, best_W, best_err
    
    # Try decreasing ranks
    # Lower bound: at least max(dim_u, dim_v, dim_w) is not quite right
    # For matrix multiplication n×m×p, the rank is at least max(n*m, m*p, p*n) in some cases
    # But Strassen shows we can do better than n*m*p
    # Let's try from a reasonable starting point
    
    # The schoolbook rank is n*m*p = 60
    # Current best is 48
    # Try to find something lower
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, create schoolbook as fallback
    U_sb = np.zeros((dim_u, n * m * p))
    V_sb = np.zeros((dim_v, n * m * p))
    W_sb = np.zeros((dim_w, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = (U_sb, V_sb, W_sb, n * m * p)
    
    # Try ranks from 47 down
    for rank in range(47, max(14, 47 - 30), -1):
        if time.time() > deadline - 30:
            break
        
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            
            U, V, W, err = try_rank(rank, T, max_iters=3000, lr=0.05, seed=seed)
            
            if err < 1e-6:
                # Snap to rational
                U_snapped = snap_to_rational(U, max_denom=4)
                V_snapped = snap_to_rational(V, max_denom=4)
                W_snapped = snap_to_rational(W, max_denom=4)
                
                # Verify snapped version
                if verify_decomposition(U_snapped, V_snapped, W_snapped, tol=1e-6):
                    best_result = (U_snapped, V_snapped, W_snapped, rank)
                    break
                else:
                    # Try original (non-snapped)
                    if verify_decomposition(U, V, W, tol=1e-6):
                        best_result = (U, V, W, rank)
                        break
    
    # If we found something better, use it; otherwise use schoolbook
    if best_result is not None and best_result[3] < n * m * p:
        U, V, W, rank = best_result
    else:
        U, V, W, rank = best_result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
