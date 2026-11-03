"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def _is_half_integer(x, tol=1e-9):
    """Check if x is a half-integer (k/2 for integer k)."""
    return abs(x * 2 - round(x * 2)) < tol

def _snap_to_half_integer(x):
    """Snap x to nearest half-integer."""
    return round(x * 2) / 2.0

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _residual_fast(params, n, m, p):
    """
    params: flattened vector of U, V, W
    n, m, p: dimensions
    Returns residual vector
    """
    size_u = n * m
    size_v = m * p
    size_w = p * n
    rank = len(params) // (size_u + size_v + size_w)
    
    # Unpack and Reshape
    u_flat = params[:size_u * rank]
    v_flat = params[size_u * rank : size_u * rank + size_v * rank]
    w_flat = params[size_u * rank + size_v * rank :]
    
    U = u_flat.reshape((n, m, rank))
    V = v_flat.reshape((m, p, rank))
    W = w_flat.reshape((p, n, rank))
    
    # Compute sum over rank c: U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Result shape (n, m, p)
    S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    
    # Residual is S - 1
    return (S - 1.0).flatten()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Target ranks: Known bound is 13 for 2x3x4.
    for target_rank in range(13, best_rank):
        if time.time() > deadline - 10.0:
            break
        
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            size_u = n * m
            size_v = m * p
            size_w = p * n
            total_size = (size_u + size_v + size_w) * target_rank
            
            # Random initialization
            x0 = np.random.randn(total_size) * 1.0
            
            try:
                result = least_squares(
                    _residual_fast, 
                    x0, 
                    args=(n, m, p), 
                    method='lm', 
                    max_nfev=1000,
                    xtol=1e-12,
                    ftol=1e-12
                )
                
                if result.success or result.cost < 1e-10:
                    # Check if solution is valid
                    u_flat = result.x[:size_u * target_rank]
                    v_flat = result.x[size_u * target_rank : size_u * target_rank + size_v * target_rank]
                    w_flat = result.x[size_u * target_rank + size_v * target_rank :]
                    
                    U = u_flat.reshape((size_u, target_rank))
                    V = v_flat.reshape((size_v, target_rank))
                    W = w_flat.reshape((size_w, target_rank))
                    
                    # Check half-integer constraint
                    is_valid = True
                    for mat in [U, V, W]:
                        for row in mat:
                            for val in row:
                                if not _is_half_integer(val, 0.01):
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        if not is_valid:
                            break
                    
                    if is_valid:
                        U_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in U])
                        V_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in V])
                        W_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in W])
                        
                        if _validate(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                            best_rank = target_rank
                            best_result = {
                                "U": U_snap.tolist(),
                                "V": V_snap.tolist(),
                                "W": W_snap.tolist()
                            }
                            return best_result
                            
            except Exception:
                pass
            
    return best_result

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
