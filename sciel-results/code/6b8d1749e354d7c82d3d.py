"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 140  # 140s safety margin (evaluator kills at 150s)

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a flat array.
    T[a, b, d] = 1 if a=i*m+j, b=j*p+k, d=k*n+i for some i,j,k, else 0.
    Returns T as a 3D array of shape (n*m, m*p, p*n).
    """
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-6):
    """Check if U, V, W exactly decompose T.
    Returns True if sum_t U[a,t]*V[b,t]*W[d,t] == T[a,b,d] for all a,b,d.
    """
    # Compute the full tensor from the decomposition
    # T_decomp[a,b,d] = sum_t U[a,t]*V[b,t]*W[d,t]
    # This is a tensor contraction: einsum('at,bt,dt->abd', U, V, W)
    T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    # Check all entries
    if not np.allclose(T_decomp, T, atol=tol, rtol=0):
        return False
    return True

def _alternating_minimization(T, r, n, m, p, max_iter=500, time_budget=20.0):
    """Try to find a rank-r decomposition via alternating least squares.
    Returns (U, V, W) if found, else None.
    """
    nm, mp, pn = n * m, m * p, p * n
    T_flat = T.reshape(nm, mp, pn)
    
    # Precompute the "slices" of T for efficient solving
    # For solving W[d,:]: we need to solve M @ W[d,:]^T = T[:, :, d].ravel()
    # where M = U @ V.T (nm x mp)
    # For solving U[a,:]: we need to solve M2 @ U[a,:]^T = T[a, :, :].ravel()
    # where M2 = V @ W.T (mp x pn)
    # For solving V[b,:]: we need to solve M3 @ V[b,:]^T = T[:, b, :].ravel()
    # where M3 = U @ W.T (nm x pn)
    
    # Precompute T slices
    T_w = T_flat.transpose(2, 0, 1).reshape(pn, nm * mp)  # T_w[d, a*mp+b] = T[a,b,d]
    T_u = T_flat.transpose(0, 1, 2).reshape(nm, mp * pn)  # T_u[a, b*pn+d] = T[a,b,d]
    T_v = T_flat.transpose(1, 0, 2).reshape(mp, nm * pn)  # T_v[b, a*pn+d] = T[a,b,d]
    
    start = time.time()
    
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(nm, r) * 0.5
    V = np.random.randn(mp, r) * 0.5
    W = np.random.randn(pn, r) * 0.5
    
    for iteration in range(max_iter):
        if time.time() - start > time_budget:
            return None
        
        # Solve for W given U, V
        M = U @ V.T  # (nm, mp)
        M_flat = M.reshape(nm * mp)  # not needed, M is already 2D
        # For each d: solve M @ x = T_w[d, :] where x = W[d, :]
        # This is an overdetermined system: (nm*mp) equations, r unknowns
        # Use least squares
        W_new = np.zeros((pn, r))
        for d in range(pn):
            try:
                W_new[d, :], _, _, _ = np.linalg.lstsq(M, T_w[d, :], rcond=None)
            except:
                return None
        W = W_new
        
        # Solve for U given V, W
        M2 = V @ W.T  # (mp, pn)
        U_new = np.zeros((nm, r))
        for a in range(nm):
            try:
                U_new[a, :], _, _, _ = np.linalg.lstsq(M2, T_u[a, :], rcond=None)
            except:
                return None
        U = U_new
        
        # Solve for V given U, W
        M3 = U @ W.T  # (nm, pn)
        V_new = np.zeros((mp, r))
        for b in range(mp):
            try:
                V_new[b, :], _, _, _ = np.linalg.lstsq(M3, T_v[b, :], rcond=None)
            except:
                return None
        V = V_new
        
        # Check convergence
        if _check_decomposition(U, V, W, T, tol=1e-8):
            return U, V, W
    
    # Final check
    if _check_decomposition(U, V, W, T, tol=1e-8):
        return U, V, W
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 6, 24, 16
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook rank-48 as fallback
    rank = n * m * p  # 48
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank from 47 down
    for r in range(rank - 1, 0, -1):
        if time.time() > _DEADLINE - 10:
            break
        
        # Try multiple random restarts with different seeds
        found = False
        for attempt in range(50):
            if time.time() > _DEADLINE - 5:
                break
            
            result = _alternating_minimization(T, r, n, m, p, max_iter=300, time_budget=15.0)
            if result is not None:
                U, V, W = result
                # Verify
                if _check_decomposition(U, V, W, T, tol=1e-8):
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
        
        # If we found a valid decomposition at rank r, continue to try lower ranks
        # If not, we could stop, but let's try a few more ranks in case of numerical issues
        if not found and r < rank - 3:
            # If we've failed for several consecutive ranks, stop
            # Actually, let's just continue; the deadline guard will stop us
            pass
    
    # Convert to lists of lists for the return interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
