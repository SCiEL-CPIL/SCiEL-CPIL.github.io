"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if valid else 0."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _reconstruct(U, V, W, n, m, p, rank):
    """Reconstruct tensor from factor matrices. T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    T = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    return T

def _validate(U, V, W, n, m, p, rank):
    """Check if the decomposition is exact (within tolerance)."""
    T = _reconstruct(U, V, W, n, m, p, rank)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)

def _snap_to_half_integers(U, V, W, tol=0.01):
    """Snap values close to half-integers. Returns None if snap is not close enough."""
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0
    # Only accept if the snap didn't change things too much
    if (np.max(np.abs(U - U_s)) < tol and
        np.max(np.abs(V - V_s)) < tol and
        np.max(np.abs(W - W_s)) < tol):
        # Ensure exact half-integer representation
        U_s = np.round(U_s * 2) / 2.0
        V_s = np.round(V_s * 2) / 2.0
        W_s = np.round(W_s * 2) / 2.0
        return U_s, V_s, W_s
    return None

def _compute_error(U, V, W, n, m, p, rank, T_target):
    T = _reconstruct(U, V, W, n, m, p, rank)
    return np.sum((T - T_target)**2)

def _try_random_decomposition(n, m, p, max_rank, seed, time_limit):
    """Try to find a decomposition of rank max_rank using random init + gradient descent."""
    rng = np.random.RandomState(seed)
    T_target = _build_tensor(n, m, p)
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    start_time = time.time()
    
    # Limit attempts to avoid wasting time if no solution is found quickly
    for attempt in range(10):
        if time.time() - start_time > time_limit - 10:
            break
            
        U = rng.randn(n*m, max_rank) * 0.5
        V = rng.randn(m*p, max_rank) * 0.5
        W = rng.randn(p*n, max_rank) * 0.5
        
        # Gradient descent with adaptive learning rate
        lr = 0.01
        for step in range(1000):
            if time.time() - start_time > time_limit - 10:
                break
            
            T = _reconstruct(U, V, W, n, m, p, max_rank)
            err = np.sum((T - T_target)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-12:
                break
            
            # Compute gradients
            dT = 2 * (T - T_target)
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for c in range(max_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dT_ijk = dT[i, j, k]
                            dU[i*m+j, c] += dT_ijk * V[j*p+k, c] * W[k*n+i, c]
                            dV[j*p+k, c] += dT_ijk * U[i*m+j, c] * W[k*n+i, c]
                            dW[k*n+i, c] += dT_ijk * U[i*m+j, c] * V[j*p+k, c]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Simple learning rate decay
            if step % 100 == 0:
                lr *= 0.95
        
        # Check if we found a valid solution
        if best_err < 1e-12:
            break
    
    if best_U is not None and best_err < 1e-9:
        snapped = _snap_to_half_integers(best_U, best_V, best_W)
        if snapped is not None:
            U_s, V_s, W_s = snapped
            if _validate(U_s, V_s, W_s, n, m, p, max_rank):
                return U_s, V_s, W_s
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    # Ensure these are integers (or floats that are exactly integers)
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    time_limit = 140  # Leave margin before 150s kill
    
    # Try ranks from lower to higher
    # We limit the search to a few ranks to ensure we have time to return a valid result
    # The schoolbook rank is 24. We try 23, 22, etc.
    for rank in range(schoolbook_rank - 1, max(14, schoolbook_rank - 5), -1):
        if time.time() - start_time > time_limit - 20:
            break
        
        # Use a specific time slice for each rank attempt
        rank_time_limit = min(20, time_limit - (time.time() - start_time))
        if rank_time_limit <= 5:
            break
            
        result = _try_random_decomposition(n, m, p, rank, seed=rank * 1000 + 42, time_limit=rank_time_limit)
        if result is not None:
            U_s, V_s, W_s = result
            # Convert to lists of floats/ints
            # Ensure values are exactly representable
            U_list = [[val for val in row] for row in U_s.tolist()]
            V_list = [[val for val in row] for row in V_s.tolist()]
            W_list = [[val for val in row] for row in W_s.tolist()]
            
            # Final validation of the list structure
            if _validate(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank):
                best_result = {
                    "U": U_list,
                    "V": V_list,
                    "W": W_list
                }
                best_rank = rank
                break  # Found a valid lower rank
    
    return best_result
# EVOLVE-BLOCK-END
