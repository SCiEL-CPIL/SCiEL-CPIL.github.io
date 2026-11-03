import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a better decomposition
    # Target rank: rank_schoolbook - 1
    target_rank = rank_schoolbook - 1
    if target_rank < 1:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    start_time = time.time()
    time_limit = 10.0  # seconds
    
    # Build the target tensor T
    # T[i, k, i', j', j'', k'] = 1 if i==i' and j'==j'' and k==k'
    # We flatten the indices for the linear system
    # Variables: W[j, k, r]
    # Equations: sum_r U[i, k, r] V[i', j', r] W[j'', k', r] = T[i, k, i', j', j'', k']
    
    # We will use ALS on U, V, W
    # Initialize randomly
    best_U, best_V, best_W = None, None, None
    best_error = float('inf')
    
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(42 + restart)
        # Random initialization
        U = np.random.randn(n * p, target_rank) * 0.1
        V = np.random.randn(n * m, target_rank) * 0.1
        W = np.random.randn(m * p, target_rank) * 0.1
        
        # Normalize to prevent scale issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Solve for W
            # For each j, k, we have a vector W[j*p+k, :]
            # The equations involving W[j'', k', :] are for all i, i', j'
            # sum_r U[i, k, r] V[i', j', r] W[j'', k', r] = T[i, k, i', j', j'', k']
            # Let x_r = W[j'', k', r]
            # sum_r (U[i, k, r] V[i', j', r]) x_r = T[i, k, i', j', j'', k']
            # This is a linear system A x = b for each (j'', k')
            # A is of size (n * n * m) x target_rank?
            # Indices of A: (i, i', j')
            # A[i, i', j', r] = U[i, k, r] V[i', j', r]
            # b[i, i', j'] = T[i, k, i', j', j'', k']
            
            # We can solve this for all (j'', k') at once or one by one.
            # Let's do one by one for simplicity and speed check.
            for j in range(m):
                for k in range(p):
                    idx_w = j * p + k
                    # Construct A and b
                    # A has rows for (i, i', j')
                    # i in 0..n-1, i' in 0..n-1, j' in 0..m-1
                    # Total rows: n * n * m
                    A_mat = np.zeros((n * n * m, target_rank))
                    b_vec = np.zeros(n * n * m)
                    row = 0
                    for i in range(n):
                        for ip in range(n):
                            for jp in range(m):
                                # T[i, k, ip, jp, j, k] is 1 if i==ip and jp==j and k==k
                                # Here k is fixed from loop, so k==k is true.
                                # So T is 1 if i==ip and jp==j.
                                val = 1.0 if (i == ip and jp == j) else 0.0
                                b_vec[row] = val
                                # A[row, r] = U[i, k, r] * V[ip, jp, r]
                                # U index: i, k -> i*p + k
                                # V index: ip, jp -> ip*m + jp
                                A_mat[row, :] = U[i * p + k, :] * V[ip * m + jp, :]
                                row += 1
                    # Solve A_mat x = b_vec
                    # Use least squares
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[idx_w, :] = x
                    except:
                        pass
                    
            # Solve for U
            # For each i, k, vector U[i, k, :]
            # Equations: sum_r U[i, k, r] V[i', j', r] W[j'', k', r] = T[i, k, i', j', j'', k']
            # x_r = U[i, k, r]
            # A[i', j', j'', k', r] = V[i', j', r] W[j'', k', r]
            # b[i', j', j'', k'] = T[i, k, i', j', j'', k']
            for i in range(n):
                for k in range(p):
                    idx_u = i * p + k
                    A_mat = np.zeros((n * m * m * p, target_rank))
                    b_vec = np.zeros(n * m * m * p)
                    row = 0
                    for ip in range(n):
                        for jp in range(m):
                            for jj in range(m):
                                for kk in range(p):
                                    # T[i, k, ip, jp, jj, kk] = 1 if i==ip and jp==jj and k==kk
                                    val = 1.0 if (i == ip and jp == jj and k == kk) else 0.0
                                    b_vec[row] = val
                                    A_mat[row, :] = V[ip * m + jp, :] * W[jj * p + kk, :]
                                    row += 1
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[idx_u, :] = x
                    except:
                        pass
                        
            # Solve for V
            # For each i, j, vector V[i, j, :]
            # Equations: sum_r U[i, k, r] V[i', j', r] W[j'', k', r] = T[i, k, i', j', j'', k']
            # x_r = V[i', j', r]
            # A[i, k, j'', k', r] = U[i, k, r] W[j'', k', r]
            # b[i, k, j'', k'] = T[i, k, i', j', j'', k']
            for ip in range(n):
                for jp in range(m):
                    idx_v = ip * m + jp
                    A_mat = np.zeros((n * p * m * p, target_rank))
                    b_vec = np.zeros(n * p * m * p)
                    row = 0
                    for i in range(n):
                        for k in range(p):
                            for jj in range(m):
                                for kk in range(p):
                                    # T[i, k, ip, jp, jj, kk] = 1 if i==ip and jp==jj and k==kk
                                    val = 1.0 if (i == ip and jp == jj and k == kk) else 0.0
                                    b_vec[row] = val
                                    A_mat[row, :] = U[i * p + k, :] * W[jj * p + kk, :]
                                    row += 1
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[idx_v, :] = x
                    except:
                        pass
                        
        # Check error
        error = 0.0
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for jj in range(m):
                            for kk in range(p):
                                s = 0.0
                                for r in range(target_rank):
                                    s += U[i * p + k, r] * V[ip * m + jp, r] * W[jj * p + kk, r]
                                val = 1.0 if (i == ip and jp == jj and k == kk) else 0.0
                                error += (s - val) ** 2
                                
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_error < 1e-9:
            break

    # Check if best solution is exact
    if best_error < 1e-9 and best_U is not None:
        # Convert to lists
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
