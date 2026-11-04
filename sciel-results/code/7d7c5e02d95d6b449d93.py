"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import sys
import numpy as np
import random

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_flat(T, n, m, p):
    """Flatten the 3D tensor to a 1D vector using the index mapping."""
    return T.reshape(n*m*p)

def _flat_to_tensor(flat, n, m, p):
    """Reshape flat vector back to 3D tensor."""
    return flat.reshape(n, m, p)

def _decompose_tensor(T, n, m, p, rank, max_iter=500, tol=1e-10, seed=42):
    """
    Find a rank decomposition of the tensor T using iterative refinement.
    
    The tensor T[i,j,k] = 1 for all valid (i,j,k) in matrix multiplication.
    We seek U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rng = np.random.default_rng(seed)
    
    # Initialize with random values
    U = rng.standard_normal((n*m, rank))
    V = rng.standard_normal((m*p, rank))
    W = rng.standard_normal((p*n, rank))
    
    # Normalize to avoid numerical issues
    for r in range(rank):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        if norm_u > 1e-12 and norm_v > 1e-12 and norm_w > 1e-12:
            scale = (norm_u * norm_v * norm_w) ** (1/3)
            U[:, r] /= norm_u
            V[:, r] /= norm_v
            W[:, r] /= norm_w
            U[:, r] *= scale
            V[:, r] *= scale
            W[:, r] *= scale
    
    T_flat = _tensor_to_flat(T, n, m, p)
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                idx = i*m + j
                # For each r, U[idx, r] = sum_k T[i,j,k] * V[j*p+k, r] * W[k*n+i, r] / sum_r' V[j*p+k, r'] * W[k*n+i, r']
                # But we need to do this for each r independently
                for r in range(rank):
                    numerator = 0.0
                    denominator = 0.0
                    for k in range(p):
                        num = T[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                        denom = 0.0
                        for r2 in range(rank):
                            denom += V[j*p+k, r2] * W[k*n+i, r2]
                        if abs(denom) > 1e-15:
                            numerator += T[i, j, k] * V[j*p+k, r] * W[k*n+i, r] / denom
                        else:
                            numerator += T[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                    U[idx, r] = numerator
        
        # Update V
        for j in range(m):
            for k in range(p):
                idx = j*p + k
                for r in range(rank):
                    numerator = 0.0
                    for i in range(n):
                        num = T[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
                        denom = 0.0
                        for r2 in range(rank):
                            denom += U[i*m+j, r2] * W[k*n+i, r2]
                        if abs(denom) > 1e-15:
                            numerator += T[i, j, k] * U[i*m+j, r] * W[k*n+i, r] / denom
                        else:
                            numerator += T[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
                    V[idx, r] = numerator
        
        # Update W
        for k in range(p):
            for i in range(n):
                idx = k*n + i
                for r in range(rank):
                    numerator = 0.0
                    for j in range(m):
                        num = T[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
                        denom = 0.0
                        for r2 in range(rank):
                            denom += U[i*m+j, r2] * V[j*p+k, r2]
                        if abs(denom) > 1e-15:
                            numerator += T[i, j, k] * U[i*m+j, r] * V[j*p+k, r] / denom
                        else:
                            numerator += T[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
                    W[idx, r] = numerator
        
        # Compute residual
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residual += (s - T[i, j, k])**2
        residual = np.sqrt(residual)
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if residual < tol:
            break
    
    return best_U, best_V, best_W, best_residual

def _try_rank(n, m, p, rank, max_restarts=5, time_limit=20):
    """Try to find a rank decomposition of the given rank."""
    T = _build_tensor(n, m, p)
    start_time = time.time()
    
    best_residual = np.inf
    best_result = None
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        seed = 42 + restart * 1000
        U, V, W, residual = _decompose_tensor(T, n, m, p, rank, max_iter=200, tol=1e-10, seed=seed)
        
        if residual < best_residual:
            best_residual = residual
            best_result = (U, V, W)
        
        if residual < 1e-9:
            break
    
    if best_result is not None and best_residual < 1e-9:
        U, V, W = best_result
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But the actual rank may be higher. Let's try from low to high.
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # For 2x3x7, let's try ranks from 15 to 41
    for rank in range(15, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(n, m, p, rank, max_restarts=3, time_limit=15)
        if result is not None:
            best = result
            best_rank = rank
            break  # Found a valid decomposition, this is the minimum
    
    return best
# EVOLVE-BLOCK-END
