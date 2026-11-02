"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if valid else 0."""
    return np.ones((n, m, p))

def _reconstruct(U, V, W, n, m, p, rank):
    """Reconstruct tensor from factor matrices. T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    # Use einsum for speed
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We need T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This is a contraction over c.
    # Let's use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return T

def _validate(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if the decomposition is exact (within tolerance)."""
    T = _reconstruct(U, V, W, n, m, p, rank)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

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
    # We will use a smarter initialization: start with schoolbook and remove random columns?
    # Or just random init.
    # The prompt suggests "geometric invariant theory approach... random linear reparametrizations".
    # This is complex. Let's stick to improved random init + GD with better convergence.
    
    for attempt in range(10):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Random initialization
        U = rng.randn(n*m, max_rank) * 0.5
        V = rng.randn(m*p, max_rank) * 0.5
        W = rng.randn(p*n, max_rank) * 0.5
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        for step in range(500):
            if time.time() - start_time > time_limit - 10:
                break
            
            T = _reconstruct(U, V, W, n, m, p, max_rank)
            err = np.sum((T - T_target)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-12:
                break
            
            # Compute gradients using einsum for speed
            dT = 2 * (T - T_target)
            
            # dU[i*m+j, c] = sum_{k} dT[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
            # Reshape dT to (n, m, p)
            dT_r = dT.reshape(n, m, p)
            
            # V_r is (m, p, rank), W_r is (p, n, rank)
            V_r = V.reshape(m, p, max_rank)
            W_r = W.reshape(p, n, max_rank)
            U_r = U.reshape(n, m, max_rank)
            
            # dU_r[i,j,c] = sum_k dT_r[i,j,k] * V_r[j,k,c] * W_r[k,i,c]
            # einsum: 'ijk,jkc,kic->ijc'
            dU_r = np.einsum('ijk,jkc,kic->ijc', dT_r, V_r, W_r)
            
            # dV_r[j,k,c] = sum_i dT_r[i,j,k] * U_r[i,j,c] * W_r[k,i,c]
            # einsum: 'ijk,ijc,kic->jkc'
            dV_r = np.einsum('ijk,ijc,kic->jkc', dT_r, U_r, W_r)
            
            # dW_r[k,i,c] = sum_j dT_r[i,j,k] * U_r[i,j,c] * V_r[j,k,c]
            # einsum: 'ijk,ijc,jkc->kic'
            dW_r = np.einsum('ijk,ijc,jkc->kic', dT_r, U_r, V_r)
            
            dU = dU_r.reshape(n*m, max_rank)
            dV = dV_r.reshape(m*p, max_rank)
            dW = dW_r.reshape(p*n, max_rank)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Simple learning rate decay
            if step % 50 == 0:
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
    # Known lower bound is 12.
    # Let's try a few more ranks than the seed, but keep it safe.
    # Seed tried range(23, 19, -1) i.e., 23, 22, 21, 20.
    # Let's try 23 down to 18.
    
    for rank in range(schoolbook_rank - 1, max(17, schoolbook_rank - 6), -1):
        if time.time() - start_time > time_limit - 20:
            break
        
        # Use a specific time slice for each rank attempt
        rank_time_limit = min(15, time_limit - (time.time() - start_time))
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
