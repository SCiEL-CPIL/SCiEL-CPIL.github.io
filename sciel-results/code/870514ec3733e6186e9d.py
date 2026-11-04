"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_target_tensor(n, m, p):
    """
    Builds the target tensor T of shape (n*m, m*p, p*n).
    T[I, J, K] = 1 if I=i*m+j, J=j*p+k, K=k*n+i for some i,j,k.
    Otherwise 0.
    """
    size_u = n * m
    size_v = m * p
    size_w = p * n
    T = np.zeros((size_u, size_v, size_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    return T

def _reconstruct(U, V, W):
    """
    Reconstructs the tensor from factor matrices.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Returns tensor of shape (n*m, m*p, p*n).
    """
    r = U.shape[1]
    # Efficient reconstruction using einsum or loops
    # T[I, J, K] = sum_c U[I, c] * V[J, c] * W[K, c]
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    return T

def _error(U, V, W, T):
    R = _reconstruct(U, V, W)
    return np.linalg.norm(R - T, 'fro')

def _snap_to_half_integer(M):
    """
    Snaps matrix M to nearest half-integer (0.5 * integer).
    """
    return np.round(M * 2.0) / 2.0

def _verify(U, V, W, T):
    """
    Verifies if the decomposition is exact (within 1e-9).
    """
    R = _reconstruct(U, V, W)
    return np.allclose(R, T, atol=1e-9)

def _try_rank(n, m, p, rank, T, time_limit):
    """
    Attempts to find a decomposition of given rank.
    """
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Initial guess: Random
    # We can try multiple restarts
    best_result = None
    best_err = float('inf')
    
    num_restarts = 5
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Initialize U, V, W
        # Scale: entries in T are 1. 
        # If we use random init, scale should be such that U*V*W ~ 1.
        # Let's try U ~ N(0, 1/sqrt(rank)), etc.
        scale = 1.0 / np.sqrt(rank)
        U = np.random.randn(size_u, rank) * scale
        V = np.random.randn(size_v, rank) * scale
        W = np.random.randn(size_w, rank) * scale
        
        # Flatten for optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat, V_flat, W_flat = x[:size_u*rank], x[size_u*rank:size_u*rank+size_v*rank], x[size_u*rank+size_v*rank:]
            U_mat = U_flat.reshape(size_u, rank)
            V_mat = V_flat.reshape(size_v, rank)
            W_mat = W_flat.reshape(size_w, rank)
            R = _reconstruct(U_mat, V_mat, W_mat)
            diff = R - T
            return np.sum(diff * diff)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < best_err:
                best_err = res.fun
                # Extract matrices
                U_flat, V_flat, W_flat = res.x[:size_u*rank], res.x[size_u*rank:size_u*rank+size_v*rank], res.x[size_u*rank+size_v*rank:]
                U_mat = U_flat.reshape(size_u, rank)
                V_mat = V_flat.reshape(size_v, rank)
                W_mat = W_flat.reshape(size_w, rank)
                
                # Snap to half-integers
                U_snap = _snap_to_half_integer(U_mat)
                V_snap = _snap_to_half_integer(V_mat)
                W_snap = _snap_to_half_integer(W_mat)
                
                if _verify(U_snap, V_snap, W_snap, T):
                    return U_snap, V_snap, W_snap
                else:
                    # Check if the non-snapped version is good enough to snap differently?
                    # Or just store the best non-snapped for now
                    if best_result is None or res.fun < best_result[0]:
                        best_result = (res.fun, U_mat.copy(), V_mat.copy(), W_mat.copy())
        except Exception as e:
            pass
            
    # If we found a good non-snapped solution, try snapping again with different rounding?
    # Or just return None if no exact snapped solution found.
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    start_time = time.time()
    total_time_limit = 100.0 # Leave 20s for safety
    
    # Try ranks from 20 to 23
    for rank in [20, 21, 22, 23]:
        if time.time() - start_time > total_time_limit:
            break
        res = _try_rank(n, m, p, rank, T, total_time_limit - (time.time() - start_time))
        if res is not None:
            U, V, W = res
            # Convert to lists for return
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
