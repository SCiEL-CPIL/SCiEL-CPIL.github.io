import time
import numpy as np
from scipy.optimize import least_squares

def _strassen_222():
    """Return the 7 rank-1 terms of the 2x2x2 Strassen decomposition."""
    return [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]

def _construct_disjoint(n, m, p):
    """Construct the known rank-88 disjoint decomposition."""
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    col = 0
    strassen_terms = _strassen_222()
    
    # Disjoint blocks: 
    # i: 0,1 (Strassen), 2 (School)
    # j: 0,1 (Strassen), 2,3 (Strassen)
    # k: 0,1 (Strassen), 2,3 (Strassen), 4,5 (Strassen), 6,7 (Strassen)
    
    # Partitions
    part_n = [ [0,1], [2] ]
    part_m = [ [0,1], [2,3] ]
    part_p = [ [0,1], [2,3], [4,5], [6,7] ]
    
    for bn in part_n:
        for bm in part_m:
            for bp in part_p:
                sn, sm, sp = len(bn), len(bm), len(bp)
                
                if sn == 2 and sm == 2 and sp == 2:
                    i0, j0, k0 = bn[0], bm[0], bp[0]
                    for (u, v, w) in strassen_terms:
                        for di in range(2):
                            for dj in range(2):
                                U[(i0+di)*m + (j0+dj), col] = u[di*2+dj]
                        for dj in range(2):
                            for dk in range(2):
                                V[(j0+dj)*p + (k0+dk), col] = v[dj*2+dk]
                        for di in range(2):
                            for dk in range(2):
                                W[(k0+dk)*n + (i0+di), col] = w[di*2+dk]
                        col += 1
                else:
                    # Schoolbook
                    for i in bn:
                        for j in bm:
                            for k in bp:
                                U[i*m + j, col] = 1
                                V[j*p + k, col] = 1
                                W[k*n + i, col] = 1
                                col += 1
                                
    return col, U[:, :col], V[:, :col], W[:, :col]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact."""
    U, V, W = np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_remove_term(U, V, W, n, m, p, term_idx, deadline):
    """
    Try to remove term 'term_idx' by optimizing the remaining terms.
    Returns (True, U_new, V_new, W_new) if successful, else (False, None, None, None).
    """
    r = U.shape[1]
    if term_idx < 0 or term_idx >= r:
        return False, None, None, None
    
    # Create a mask for all terms except term_idx
    mask = np.ones(r, dtype=bool)
    mask[term_idx] = False
    
    U_sub = U[:, mask]
    V_sub = V[:, mask]
    W_sub = W[:, mask]
    
    r_sub = U_sub.shape[1]
    
    # Flatten for optimization
    x0 = np.concatenate([U_sub.ravel(), V_sub.ravel(), W_sub.ravel()])
    
    # Pre-compute tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def residuals(x):
        U_flat, V_flat, W_flat = x[:n*m*r_sub], x[n*m*r_sub:2*n*m*r_sub], x[2*n*m*r_sub:]
        U_mat = U_flat.reshape(n*m, r_sub)
        V_mat = V_flat.reshape(m*p, r_sub)
        W_mat = W_flat.reshape(p*n, r_sub)
        
        T_sum = np.einsum('it,jt,kt->ijk', U_mat, V_mat, W_mat)
        res = T - T_sum
        return res.ravel()
        
    try:
        # Use a tight budget for this local search
        # max_nfev should be small to allow many attempts
        result = least_squares(residuals, x0, method='trf', max_nfev=200, xtol=1e-12, ftol=1e-12)
        
        if result.success and np.max(np.abs(result.fun)) < 1e-8:
            x_opt = result.x
            U_opt = x_opt[:n*m*r_sub].reshape(n*m, r_sub)
            V_opt = x_opt[n*m*r_sub:2*n*m*r_sub].reshape(m*p, r_sub)
            W_opt = x_opt[2*n*m*r_sub:].reshape(p*n, r_sub)
            
            # Snap to half-integers
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            # Verify
            if _verify(U_snap, V_snap, W_snap, n, m, p):
                # Re-insert the removed term as zeros to maintain shape? 
                # No, we return the reduced matrices.
                return True, U_snap, V_snap, W_snap
    except Exception:
        pass
        
    return False, None, None, None

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 130
    
    # 1. Get the known disjoint decomposition (Rank 88)
    rank_88, U_88, V_88, W_88 = _construct_disjoint(n, m, p)
    if not _verify(U_88, V_88, W_88, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    W[k*n+i][c] = 1.0
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank_88
    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
    
    # 2. Greedy Rank Reduction
    # We try to remove terms one by one.
    # We iterate multiple passes in case removing one term enables removing another.
    current_U = U_88.copy()
    current_V = V_88.copy()
    current_W = W_88.copy()
    current_rank = rank_88
    
    # We will try to remove terms in a random order to escape local minima
    np.random.seed(42)
    
    # Pass 1: Try to remove each term once
    for _pass in range(2): # 2 passes
        if time.time() > deadline:
            break
            
        # Get current indices
        r = current_U.shape[1]
        indices = np.arange(r)
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            # Try to remove term at 'idx'
            success, U_new, V_new, W_new = _try_remove_term(
                current_U, current_V, current_W, n, m, p, idx, deadline
            )
            
            if success:
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_result = {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
                    
    return best_result
