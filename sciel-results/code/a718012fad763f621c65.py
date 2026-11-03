"""Structural search for sub-schoolbook rank decomposition of M_{3,4,5}."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor as a flat array.
    
    T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    T[a,b,c] = 1 iff j==j', i==i', k==k'
    """
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Verify decomposition using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = _build_target_tensor(n, m, p)
    
    return np.allclose(T_computed, T_target, atol=1e-9)

def _snap_to_half_int(arr):
    """Snap array values to nearest multiple of 0.5."""
    return np.round(arr * 2) / 2

def _try_find_decomposition(target_rank, n, m, p, deadline, max_restarts=20):
    """Try to find a decomposition with given rank using numerical optimization."""
    T_target = _build_target_tensor(n, m, p)
    T_flat = T_target.flatten()
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    best_loss = np.inf
    best_params = None
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize with schoolbook-like structure + noise
        # We use the first 'target_rank' basis vectors of the schoolbook decomposition
        # and add noise to allow movement.
        
        U_init = np.zeros((size_u, target_rank))
        V_init = np.zeros((size_v, target_rank))
        W_init = np.zeros((size_w, target_rank))
        
        # Fill with schoolbook terms
        count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if count >= target_rank:
                        break
                    U_init[i * m + j, count] = 1.0
                    V_init[j * p + k, count] = 1.0
                    W_init[k * n + i, count] = 1.0
                    count += 1
                    if count >= target_rank:
                        break
                if count >= target_rank:
                    break
            if count >= target_rank:
                break
        
        # Add noise
        noise_level = 0.5
        U_init += np.random.normal(0, noise_level, U_init.shape)
        V_init += np.random.normal(0, noise_level, V_init.shape)
        W_init += np.random.normal(0, noise_level, W_init.shape)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(x):
            if time.time() > deadline - 2:
                return np.zeros(len(T_flat)) # Early exit hack? No, just return large error
                # Actually, least_squares doesn't handle early exit well. 
                # We rely on the deadline check in the outer loop.
                
            U = x[:size_u * target_rank].reshape(size_u, target_rank)
            V = x[size_u * target_rank : size_u * target_rank + size_v * target_rank].reshape(size_v, target_rank)
            W = x[size_u * target_rank + size_v * target_rank :].reshape(size_w, target_rank)
            
            T_computed = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_computed - T_target).flatten()
        
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            loss = result.cost * 2 # cost is 0.5 * sum(residuals^2)
            
            if loss < best_loss:
                best_loss = loss
                best_params = result.x
                
                # Check if this solution is close to half-integers
                U_sol = best_params[:size_u * target_rank].reshape(size_u, target_rank)
                V_sol = best_params[size_u * target_rank : size_u * target_rank + size_v * target_rank].reshape(size_v, target_rank)
                W_sol = best_params[size_u * target_rank + size_v * target_rank :].reshape(size_w, target_rank)
                
                # Try snapping
                U_snap = _snap_to_half_int(U_sol)
                V_snap = _snap_to_half_int(V_sol)
                W_snap = _snap_to_half_int(W_sol)
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), target_rank
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition
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
    
    result = {"U": U, "V": V, "W": W}
    
    # Try to find better decompositions
    # Try ranks 58, 59
    for target_rank in [58, 59]:
        if time.time() > deadline - 10:
            break
            
        solution = _try_find_decomposition(target_rank, n, m, p, deadline, max_restarts=10)
        if solution:
            U_sol, V_sol, W_sol, r = solution
            result = {"U": U_sol, "V": V_sol, "W": W_sol}
            # Update result to the better one
            # We don't break immediately in case we find 58 after 59? 
            # We try 58 first, so if we find 58, we can stop.
            if target_rank == 58:
                break
            # If we found 59, we continue to see if we can find 58?
            # The loop order is [58, 59]. So if we find 58, we break.
            # If we don't find 58, we try 59. If we find 59, we update result.
            # Then loop ends.
            
    return result
