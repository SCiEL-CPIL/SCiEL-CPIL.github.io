import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def _verify_solution_int(U_int, V_int, W_int, n, m, p):
    """
    Verifies if U_int, V_int, W_int (integers, representing 2x values) 
    form a valid decomposition.
    Condition: sum_c (U_int/2) (V_int/2) (W_int/2) = delta
    => sum_c U_int V_int W_int = 8 * delta
    """
    rank = U_int.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0
                    for c in range(rank):
                        s += U_int[i * m + j, c] * V_int[jp * p + k, c] * W_int[k * n + i, c]
                    target = 8 if j == jp else 0
                    if s != target:
                        return False
    return True

def _snap_and_verify(U, V, W, n, m, p):
    """
    Snaps U, V, W to half-integers and verifies.
    Returns (U_list, V_list, W_list) if valid, else None.
    """
    # Snap to nearest half-integer
    # x -> round(2*x)/2
    U_half = np.round(U * 2) / 2.0
    V_half = np.round(V * 2) / 2.0
    W_half = np.round(W * 2) / 2.0
    
    # Convert to integers for exact verification
    U_int = np.round(U * 2).astype(np.int64)
    V_int = np.round(V * 2).astype(np.int64)
    W_int = np.round(W * 2).astype(np.int64)
    
    if _verify_solution_int(U_int, V_int, W_int, n, m, p):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try snapping to integers as well
    U_int2 = np.round(U).astype(np.int64)
    V_int2 = np.round(V).astype(np.int64)
    W_int2 = np.round(W).astype(np.int64)
    
    # If integers work, the condition is sum U V W = delta.
    # My _verify_solution_int expects sum = 8 * delta.
    # So if we use integers directly, we need to adjust the check.
    # Let's just check the float version for integers.
    # Or just return if float check passes?
    # Better: check if U_int2, V_int2, W_int2 satisfy sum U V W = delta.
    # We can reuse _verify_solution_int by passing 2*U_int2?
    # No, let's just do a float check for the integer case to be safe, or adjust verify.
    
    # For simplicity, if integer snap fails the 8x check, try float check
    # Actually, if U is integer, U_int = 2U. Then sum U_int V_int W_int = 8 sum U V W.
    # So if sum U V W = delta, then sum U_int V_int W_int = 8 delta.
    # So the same _verify_solution_int works!
    
    if _verify_solution_int(U_int2, V_int2, W_int2, n, m, p):
        return U_int2.astype(float).tolist(), V_int2.astype(float).tolist(), W_int2.astype(float).tolist()
        
    return None

def _residual_func(x, n, m, p, rank):
    """
    Vectorized residual function for least_squares.
    """
    U_flat = x[:n*m*rank].reshape(n*m, rank)
    V_flat = x[n*m*rank:(n*m*rank + m*p*rank)].reshape(m*p, rank)
    W_flat = x[(n*m*rank + m*p*rank):].reshape(p*n, rank)
    
    residuals = []
    # Pre-allocate or append
    # n=2, m=3, p=4
    # i in 0..1, k in 0..3, j in 0..2, jp in 0..2
    
    for i in range(n):
        for k in range(p):
            # Slice U for fixed i: rows i*m to i*m+m-1
            U_slice = U_flat[i*m : (i+1)*m, :] # shape (m, rank)
            # Slice V for fixed k: rows 0*p+k, 1*p+k, ... (m-1)*p+k
            V_slice = V_flat[::p, k] # This is wrong. V index is jp*p+k.
            # V_flat is (m*p, rank). Indices: 0, 1, 2, 3, 4, 5, 6, 7...
            # jp=0 -> k. jp=1 -> p+k. jp=2 -> 2p+k.
            V_slice = np.empty((m, rank))
            for jp in range(m):
                V_slice[jp] = V_flat[jp*p + k, :]
            
            # W vector for fixed k, i
            W_vec = W_flat[k*n + i, :] # shape (rank,)
            
            # Compute M = (U_slice * W_vec) @ V_slice.T
            # U_slice * W_vec broadcasts (m, rank) * (rank,) -> (m, rank)
            UW = U_slice * W_vec
            M = UW @ V_slice.T # shape (m, m)
            
            # Residuals for this i, k
            # Target is identity matrix of size m
            # Residual = M - I
            I = np.eye(m)
            residuals.extend((M - I).flatten().tolist())
            
    return np.array(residuals)

def _find_decomposition(rank, n, m, p, max_restarts=50, timeout=20):
    start_time = time.time()
    num_params = (n*m + m*p + p*n) * rank
    
    for r_idx in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        x0 = np.random.randn(num_params) * 0.5
        
        try:
            result = least_squares(_residual_func, x0, args=(n, m, p, rank), method='trf', 
                                   xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                   max_nfev=2000)
            
            if result.success or result.cost < 1e-10:
                U = result.x[:n*m*rank].reshape(n*m, rank)
                V = result.x[n*m*rank:(n*m*rank + m*p*rank)].reshape(m*p, rank)
                W = result.x[(n*m*rank + m*p*rank):].reshape(p*n, rank)
                
                snapped = _snap_and_verify(U, V, W, n, m, p)
                if snapped is not None:
                    return snapped
                    
        except Exception:
            continue
            
    return None

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
    n, m, p = DIMENSIONS
    deadline = 140  # Leave 10s margin
    
    # Try ranks from 23 down to 19
    # Lower ranks are harder, but if we find one, it's a better score.
    # Start with 23 (most likely to be found if < 24)
    for r in range(23, 18, -1):
        if time.time() > deadline - 15:
            break
            
        # Allocate time based on remaining time
        # Give 20s to each rank attempt
        timeout = 20
        if time.time() + timeout > deadline - 5:
            timeout = deadline - time.time() - 5
            if timeout < 2:
                break
                
        result = _find_decomposition(r, n, m, p, max_restarts=100, timeout=timeout)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
