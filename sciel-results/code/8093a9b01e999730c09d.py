"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    def build_tensor_constraints():
        """Build the list of (a,b,c,expected) for the multiplication tensor."""
        constraints = []
        for i in range(n):
            for j in range(m):
                for jp in range(m):  # j'
                    for k in range(p):
                        for kp in range(p):  # k'
                            for ip in range(n):  # i'
                                a = i * m + j
                                b = jp * p + k
                                c = kp * n + ip
                                # T=1 iff i==i' and j==jp and k==kp
                                if i == ip and j == jp and k == kp:
                                    constraints.append((a, b, c, 1.0))
                                else:
                                    constraints.append((a, b, c, 0.0))
        return constraints
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U,V,W form a valid decomposition."""
        n, m, p = DIMENSIONS
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                a = i * m + j
                                b = jp * p + k
                                c = kp * n + ip
                                s = 0.0
                                for r in range(U.shape[1]):
                                    s += U[a, r] * V[b, r] * W[c, r]
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                err = abs(s - expected)
                                if err > max_err:
                                    max_err = err
        return max_err
    
    def try_rank(rank, seed=0):
        """Try to find a rank-r decomposition using iterative methods."""
        np.random.seed(seed)
        n, m, p = DIMENSIONS
        # Initialize randomly
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Use alternating least squares / gradient-based approach
        # The objective is sum of (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
        # This is non-convex, so we try multiple restarts
        
        best_U, best_V, best_W, best_err = None, None, None, float('inf')
        
        constraints = build_tensor_constraints()
        
        for restart in range(200):
            if time.time() > deadline - 5:
                break
            np.random.seed(seed + restart)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Alternating minimization: fix two, optimize the third
            for it in range(500):
                if time.time() > deadline - 5:
                    break
                    
                # Optimize U: for each a, the contribution to error involves U[a,:]
                # Error term: for each (a,b,c), (U[a,:] · (V[b,:] * W[c,:]) - T)^2
                # Gradient wrt U[a,r]: 2 * sum_{b,c} (V[b,r]*W[c,r]) * (dot - T)
                # This is complex; use a simpler approach
                
                # Actually, let's use a direct approach: for fixed V,W, 
                # U[a,r] appears linearly in each term where a is fixed.
                # The optimal U[a,:] minimizes sum_{b,c} (U[a,:]·(V[b,:]*W[c,:]) - T[a,b,c])^2
                # This is a linear least squares problem!
                
                # For each a: minimize ||M_a @ U[a,:] - t_a||^2
                # where M_a[b,c] = V[b,:]*W[c,:] (element-wise product, flattened)
                # t_a[b,c] = T[a,b,c]
                
                # Build M for each a
                for a in range(n * m):
                    # M has rows indexed by (b,c) pairs
                    M_rows = []
                    t_vec = []
                    for b in range(m * p):
                        for c in range(p * n):
                            M_rows.append(V[b, :] * W[c, :])
                            # Determine T value
                            # a = i*m+j, b = jp*p+k, c = kp*n+ip
                            i, j = a // m, a % m
                            jp, k = b // p, b % p
                            kp, ip = c // n, c % n
                            t_val = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            t_vec.append(t_val)
                    M = np.array(M_rows)
                    t = np.array(t_vec)
                    # Solve least squares
                    U[a, :], residuals, rank_ls, sv = np.linalg.lstsq(M, t, rcond=None)
                
                # Similarly optimize V and W
                for b in range(m * p):
                    M_rows = []
                    t_vec = []
                    for a in range(n * m):
                        for c in range(p * n):
                            M_rows.append(U[a, :] * W[c, :])
                            i, j = a // m, a % m
                            jp, k = b // p, b % p
                            kp, ip = c // n, c % n
                            t_val = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            t_vec.append(t_val)
                    M = np.array(M_rows)
                    t = np.array(t_vec)
                    V[b, :], residuals, rank_ls, sv = np.linalg.lstsq(M, t, rcond=None)
                
                for c in range(p * n):
                    M_rows = []
                    t_vec = []
                    for a in range(n * m):
                        for b in range(m * p):
                            M_rows.append(U[a, :] * V[b, :])
                            i, j = a // m, a % m
                            jp, k = b // p, b % p
                            kp, ip = c // n, c % n
                            t_val = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            t_vec.append(t_val)
                    M = np.array(M_rows)
                    t = np.array(t_vec)
                    W[c, :], residuals, rank_ls, sv = np.linalg.lstsq(M, t, rcond=None)
                
                # Check error
                err = check_decomposition(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-8:
                    break
            
            if best_err < 1e-8:
                break
        
        if best_err < 1e-6:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err
    
    # Try ranks from low to high
    for rank in [7, 8, 9, 10, 11, 12]:
        if time.time() > deadline - 10:
            break
        U, V, W, err = try_rank(rank)
        if U is not None:
            # Convert to list format
            U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n * m)]
            V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m * p)]
            W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 12
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
# EVOLVE-BLOCK-END
