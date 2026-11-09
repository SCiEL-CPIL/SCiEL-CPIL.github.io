import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0

def _time_remaining():
    return _DEADLINE - time.time()

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return U, V, W

def _compute_error_sq(U, V, W, n, m, p, rank):
    """Compute the sum of squared errors."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err += (s - 1.0) ** 2
    return err

def _half_integer_penalty(x):
    """Penalty for not being half-integers."""
    # x is a flat array
    rounded = np.round(x * 2) / 2
    return np.sum((x - rounded) ** 2)

def _objective(params, n, m, p, rank):
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    U = params[:size_u].reshape(n * m, rank)
    V = params[size_u:size_u + size_v].reshape(m * p, rank)
    W = params[size_u + size_v:].reshape(p * n, rank)
    
    tensor_err = _compute_error_sq(U, V, W, n, m, p, rank)
    penalty = _half_integer_penalty(params)
    
    # We want tensor_err to be 0 and penalty to be 0.
    # If tensor_err > 0, it's invalid.
    # We prioritize finding a valid tensor first, then snapping to half-integers.
    # But snapping is hard if the solution is not close.
    # Let's use a large weight for the penalty to force lattice points.
    return tensor_err + 1000.0 * penalty

def _try_find_decomposition(n, m, p, rank, seed, time_limit):
    """
    Try to find a decomposition of given rank using Differential Evolution.
    """
    rng = np.random.RandomState(seed)
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    total_size = size_u + size_v + size_w
    
    # Bounds for differential evolution
    # We expect values to be in [-2, 2] roughly
    bounds = [(-2.0, 2.0)] * total_size
    
    def obj(x):
        return _objective(x, n, m, p, rank)

    try:
        # Differential Evolution is global and robust
        # We limit the number of iterations to fit within time_limit
        res = opt.differential_evolution(
            obj, 
            bounds, 
            seed=seed, 
            maxiter=200, 
            popsize=10, 
            tol=1e-8, 
            update_interval=1,
            polish=False # Polish with L-BFGS later if needed, but DE might be enough
        )
        
        if res.fun < 1e-6:
            U, V, W = _unflatten(res.x, n, m, p, rank)
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if _verify(U_snap, V_snap, W_snap, n, m, p, rank):
                return U_snap, V_snap, W_snap
    except Exception:
        pass
        
    return None

def _unflatten(params, n, m, p, rank):
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    U = params[:size_u].reshape(n * m, rank)
    V = params[size_u:size_u + size_v].reshape(m * p, rank)
    W = params[size_u + size_v:].reshape(p * n, rank)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition
    # Schoolbook is 60. Try 59, 58, 57
    # We allocate time to each rank attempt
    ranks_to_try = [59, 58, 57]
    
    for rank in ranks_to_try:
        if time.time() > _DEADLINE - 10: # Leave 10s for fallback
            break
            
        # Try a few seeds
        for seed in range(3):
            if time.time() > _DEADLINE - 5:
                break
            
            result = _try_find_decomposition(n, m, p, rank, seed, 30)
            if result is not None:
                U, V, W = result
                U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
