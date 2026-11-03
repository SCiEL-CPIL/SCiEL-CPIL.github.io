"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _construct_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _flatten_factors(U, V, W):
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten_factors(x, n, m, p, rank):
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U_flat = x[:size_u]
    V_flat = x[size_u:size_u + size_v]
    W_flat = x[size_u + size_v:size_u + size_v + size_w]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    return U, V, W

def _objective(x, T, n, m, p, rank, l1_weight=0.0):
    U, V, W = _unflatten_factors(x, n, m, p, rank)
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.sum((recon - T) ** 2)
    # Add L1 penalty to encourage sparsity
    if l1_weight > 0:
        err += l1_weight * (np.sum(np.abs(U)) + np.sum(np.abs(V)) + np.sum(np.abs(W)))
    return err

def _snap_and_verify(U, V, W, T, n, m, p, rank, tol=1e-6):
    """Snap values to nearest 0.5 and verify exactness."""
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
    if np.allclose(recon, T, atol=tol):
        return U_snap, V_snap, W_snap
    return None

def _try_rank_decomposition(T, target_rank, num_restarts=10, max_iter=500, tol=1e-10, l1_weight=0.01):
    n, m, p = DIMENSIONS
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_size = size_u + size_v + size_w
    
    best_err = np.inf
    best_x = None
    
    for trial in range(num_restarts):
        # Use a deterministic seed based on trial and rank for reproducibility
        np.random.seed(trial * 7919 + target_rank * 31)
        
        # Initialize with small random values
        # Sometimes it helps to initialize with schoolbook-like sparsity
        x0 = np.random.randn(total_size) * 0.5
        
        # Use L-BFGS-B
        try:
            res = minimize(_objective, x0, args=(T, n, m, p, target_rank, l1_weight), 
                           method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
                
                # Try to snap and verify early
                U, V, W = _unflatten_factors(best_x, n, m, p, target_rank)
                snapped = _snap_and_verify(U, V, W, T, n, m, p, target_rank)
                if snapped is not None:
                    return snapped[0], snapped[1], snapped[2], 0.0
        except Exception:
            continue
            
    if best_x is not None and best_err < tol:
        U, V, W = _unflatten_factors(best_x, n, m, p, target_rank)
        snapped = _snap_and_verify(U, V, W, T, n, m, p, target_rank)
        if snapped is not None:
            return snapped[0], snapped[1], snapped[2], 0.0
        # If snap fails but error is low, return raw values (evaluator might accept floats if close?)
        # Task says "integer/half-integer entries". So we MUST snap.
        # If snap fails, we didn't find a valid solution.
    return None, None, None, best_err

def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 130 # Safety margin
    
    T = _construct_tensor()
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = len(best['U'][0])
    
    # Try to find a lower rank decomposition
    # We iterate from a reasonable lower bound up to the schoolbook rank
    # Lower bound for 3x3x7 is at least max(9, 21, 21) = 21
    # Let's try from 25 to 62
    
    # We prioritize lower ranks, but give more restarts to ranks close to known bounds
    # If we find a valid decomposition, we break
    
    for target_rank in range(25, best_rank):
        if time.time() > deadline:
            break
            
        # Increase restarts for lower ranks as they are harder
        num_restarts = 5 if target_rank < 40 else 3
        
        U, V, W, err = _try_rank_decomposition(T, target_rank, num_restarts=num_restarts, max_iter=300)
        
        if U is not None:
            # Verify strictly
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(recon, T, atol=1e-9):
                best = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                break # Found minimal rank (since we iterate upwards)
                
    return best
# EVOLVE-BLOCK-END
