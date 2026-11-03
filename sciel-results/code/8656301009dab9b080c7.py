"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def _verify_half_integer(U, V, W, n, m, p):
    """Verify that U, V, W (containing integers or half-integers) form a valid decomposition."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check type constraint: entries must be integers or half-integers
    for mat in [U, V, W]:
        for val in mat.flat:
            if not np.isclose(val * 2, round(val * 2), atol=1e-9):
                return False

    # Build the target tensor T[i,j,k] = 1
    # Reconstructed: T_rec[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check if T_rec is exactly 1 (within float precision) for all i,j,k
    
    T_rec = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        
        # Vectorized update
        for i in range(n):
            for j in range(m):
                u_val = u_col[i*m+j]
                if u_val == 0: continue
                for k in range(p):
                    T_rec[i, j, k] += u_val * v_col[j*p+k] * w_col[k*n+i]
    
    # Check if all entries are 1
    return np.allclose(T_rec, 1.0, atol=1e-9)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_reduce_rank(n, m, p, target_rank, time_budget=30):
    """Try to find a decomposition of rank <= target_rank using random initialization and refinement."""
    start_time = time.time()
    
    num_rows_U = n * m
    num_rows_V = m * p
    num_rows_W = p * n
    
    # Pre-compute index mappings for vectorization
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k
    # W indices: (k, i) -> k*n + i
    
    for attempt in range(20):
        if time.time() - start_time > time_budget:
            break
        
        np.random.seed(attempt * 12345 + 67890)
        U = np.random.randn(num_rows_U, target_rank) * 0.1
        V = np.random.randn(num_rows_V, target_rank) * 0.1
        W = np.random.randn(num_rows_W, target_rank) * 0.1
        
        for iteration in range(100):
            if time.time() - start_time > time_budget:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Check convergence and snap to half-integers
            T_rec = np.zeros((n, m, p), dtype=np.float64)
            for c in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        u_val = U[i*m+j, c]
                        if u_val == 0: continue
                        for k in range(p):
                            T_rec[i, j, k] += u_val * V[j*p+k, c] * W[k*n+i, c]
            
            error = np.max(np.abs(T_rec - 1.0))
            if error < 1e-6:
                # Attempt to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify the snapped version
                if _verify_half_integer(U_snap, V_snap, W_snap, n, m, p):
                    # Convert to lists of floats (which are exact half-integers)
                    U_list = [[float(x) for x in row] for row in U_snap]
                    V_list = [[float(x) for x in row] for row in V_snap]
                    W_list = [[float(x) for x in row] for row in W_snap]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as the safe fallback
    best = _schoolbook(n, m, p)
    
    start_time = time.time()
    time_limit = 110  # Leave 10s margin for the 120s limit
    
    # Try to reduce rank. Start from a reasonable lower bound.
    # For 3x4x4, rank 48 is schoolbook.
    # We try ranks from 40 down to 30.
    for target_rank in range(40, 29, -1):
        if time.time() - start_time > time_limit:
            break
        
        result = _try_reduce_rank(n, m, p, target_rank, time_budget=10)
        if result is not None:
            best = result
            break
    
    return best
# EVOLVE-BLOCK-END
