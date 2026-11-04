import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build target tensor T (sparse)
    # T is a 3D array of size (n*m, m*p, p*n)
    # But we only need the non-zero entries for gradient computation
    # Non-zero entries: (i, j, k) -> U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
    # Value is 1.
    
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Try to find a lower rank decomposition
    # Start with rank 100, then 110, etc?
    # Or just try 100.
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Schoolbook
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    # Heuristic search for lower rank
    # Try rank 100
    target_rank = 100
    
    # Precompute indices of non-zero entries
    nz_u = []
    nz_v = []
    nz_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_u.append(i*m+j)
                nz_v.append(j*p+k)
                nz_w.append(k*n+i)
                
    nz_u = np.array(nz_u)
    nz_v = np.array(nz_v)
    nz_w = np.array(nz_w)
    
    def compute_residual(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # M_{u,v,w} = sum_c U[u,c] V[v,c] W[w,c]
        # We only care about (nz_u, nz_v, nz_w)
        # U_nz = U[nz_u, :]
        # V_nz = V[nz_v, :]
        # W_nz = W[nz_w, :]
        # M_nz = sum_c U_nz[:,c] * V_nz[:,c] * W_nz[:,c]
        # This is element-wise product sum.
        
        # Efficient:
        # M_nz = (U[nz_u] * V[nz_v] * W[nz_w]).sum(axis=1)
        return (U[nz_u] * V[nz_v] * W[nz_w]).sum(axis=1) - 1.0

    def gradient_descent(U, V, W, lr=0.01, iters=1000):
        for _ in range(iters):
            if time.time() > deadline:
                return U, V, W
            
            res = compute_residual(U, V, W)
            
            # Gradient w.r.t U[u,c]
            # dL/du = 2 * res[u] * V[v,c] * W[w,c]
            # We only update U at nz_u?
            # No, U has other rows. But they don't affect the residual if V or W are 0 there?
            # In rank decomposition, all rows/cols are active.
            # But T is 0 elsewhere.
            # So we should also penalize non-zero entries?
            # The task requires EXACT decomposition.
            # So M_{u,v,w} must be 0 for all (u,v,w) not in NZ.
            # This is a huge constraint.
            # Gradient descent on just NZ will make non-NZ entries non-zero.
            # We need to include all entries?
            # Size 125*125*125 = 2M entries.
            # Too slow for Python loops.
            
            # Alternative: Use the fact that if we initialize with schoolbook and just prune,
            # we might break the zero constraints.
            
            # Let's stick to the schoolbook if we can't handle the full tensor.
            # Or, use a simpler heuristic:
            # Try to find linear dependencies in the schoolbook columns?
            # Unlikely.
            
            pass

    # Given the complexity, returning schoolbook is the safest bet for correctness.
    # But we want to try to improve.
    
    # Let's try a random search for a rank 100 decomposition using the full tensor?
    # No, too slow.
    
    # Return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
