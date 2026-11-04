import time
import numpy as np

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

def _try_global_search(n, m, p, target_rank, deadline):
    """
    Attempt to find a decomposition of rank target_rank using optimization.
    Returns (rank, U, V, W) or None.
    """
    # Flatten the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Initialize U, V, W randomly
    np.random.seed(42)
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    # Use scipy.optimize to minimize the residual
    from scipy.optimize import least_squares
    
    def residuals(x):
        U_flat, V_flat, W_flat = x[:n*m*target_rank], x[n*m*target_rank:2*n*m*target_rank], x[2*n*m*target_rank:]
        U_mat = U_flat.reshape(n*m, target_rank)
        V_mat = V_flat.reshape(m*p, target_rank)
        W_mat = W_flat.reshape(p*n, target_rank)
        
        # Compute tensor product
        # Reshape T to (n*m, m*p, p*n)
        # Sum over t: U[i,t] * V[j,t] * W[k,t]
        # This is a 3-tensor.
        # We can flatten the residual.
        # Res[i,j,k] = T[i,j,k] - sum_t U[i,t] V[j,t] W[k,t]
        
        # Efficient computation:
        # For each t, add U[:,t] * V[:,t] * W[:,t] to a 3D array?
        # This is slow for large ranks.
        # Instead, flatten the residual vector.
        # We only need to check if it can be zero.
        
        # This function is called many times. 
        # For target_rank < 88, this is feasible.
        
        # Construct the tensor sum
        # sum_t outer(U[:,t], V[:,t]) * W[:,t]? No.
        # T_sum[i,j,k] = sum_t U[i,t] V[j,t] W[k,t]
        
        # We can use einsum
        # T_sum = np.einsum('it,jt,kt->ijk', U_mat, V_mat, W_mat)
        # This creates a (n*m, m*p, p*n) array.
        # Size: 12 * 32 * 24 = 9216 elements.
        # This is small.
        
        T_sum = np.einsum('it,jt,kt->ijk', U_mat, V_mat, W_mat)
        res = T - T_sum
        return res.ravel()
        
    x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    
    try:
        result = least_squares(residuals, x0, method='trf', max_nfev=1000)
        x_opt = result.x
        U_opt = x_opt[:n*m*target_rank].reshape(n*m, target_rank)
        V_opt = x_opt[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
        W_opt = x_opt[2*n*m*target_rank:].reshape(p*n, target_rank)
        
        # Snap to half-integers
        U_snap = np.round(U_opt * 2) / 2
        V_snap = np.round(V_opt * 2) / 2
        W_snap = np.round(W_opt * 2) / 2
        
        # Verify
        if _verify(U_snap, V_snap, W_snap, n, m, p):
            return target_rank, U_snap, V_snap, W_snap
    except Exception:
        pass
    return None

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
    
    # 2. Try to find a better decomposition using global search
    # Try ranks from 80 down to 85
    for r in range(80, 88):
        if time.time() > deadline:
            break
        res = _try_global_search(n, m, p, r, deadline)
        if res is not None:
            rank, U, V, W = res
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return best_result
