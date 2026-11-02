import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Precompute index mappings
    U_indices = []
    for i in range(n):
        for j in range(m):
            u = i * m + j
            eqs = []
            for k in range(p):
                v = j * p + k
                w = k * n + i
                eqs.append((v, w))
            U_indices.append((i, j, u, eqs))
            
    V_indices = []
    for j in range(m):
        for k in range(p):
            v = j * p + k
            eqs = []
            for i in range(n):
                u = i * m + j
                w = k * n + i
                eqs.append((u, w))
            V_indices.append((j, k, v, eqs))
            
    W_indices = []
    for k in range(p):
        for i in range(n):
            w = k * n + i
            eqs = []
            for j in range(m):
                u = i * m + j
                v = j * p + k
                eqs.append((u, v))
            W_indices.append((k, i, w, eqs))

    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        # Check exact decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[u, r] * V[v, r] * W[w, r]
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True

    def als_decompose(U_init, V_init, W_init, R, max_iters=200, tol=1e-10):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_err = np.inf
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Update U
            for i, j, u, eqs in U_indices:
                B = np.zeros((len(eqs), R))
                for idx, (v, w) in enumerate(eqs):
                    B[idx, :] = V[v, :] * W[w, :]
                try:
                    U_u, _, _, _ = np.linalg.lstsq(B, np.ones(len(eqs)), rcond=None)
                    U[u, :] = U_u
                except:
                    pass
                    
            # Update V
            for j, k, v, eqs in V_indices:
                C = np.zeros((len(eqs), R))
                for idx, (u, w) in enumerate(eqs):
                    C[idx, :] = U[u, :] * W[w, :]
                try:
                    V_v, _, _, _ = np.linalg.lstsq(C, np.ones(len(eqs)), rcond=None)
                    V[v, :] = V_v
                except:
                    pass
                    
            # Update W
            for k, i, w, eqs in W_indices:
                D = np.zeros((len(eqs), R))
                for idx, (u, v) in enumerate(eqs):
                    D[idx, :] = U[u, :] * V[v, :]
                try:
                    W_w, _, _, _ = np.linalg.lstsq(D, np.ones(len(eqs)), rcond=None)
                    W[w, :] = W_w
                except:
                    pass
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        s = 0.0
                        for r in range(R):
                            s += U[u, r] * V[v, r] * W[w, r]
                        err += (s - 1.0) ** 2
                        
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        U, V, W = best_U, best_V, best_W
        
        # Round to half-integers and verify
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        return None

    def try_rank(R, U_ref, V_ref, W_ref, max_restarts=10, time_limit=5):
        """
        Try to find a rank R solution by projecting the reference solution (rank R_ref)
        to rank R and running ALS.
        """
        R_ref = U_ref.shape[1]
        if R > R_ref:
            return None
        
        start_time = time.time()
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Create a random projection matrix P of shape (R_ref, R)
            # Using a random Gaussian matrix and orthonormalizing columns
            P = np.random.randn(R_ref, R)
            Q, _ = np.linalg.qr(P)
            # Q is (R_ref, R_ref) in full mode. Take first R columns.
            P_proj = Q[:, :R]
            
            U_init = U_ref @ P_proj
            V_init = V_ref @ P_proj
            W_init = W_ref @ P_proj
            
            res = als_decompose(U_init, V_init, W_init, R, max_iters=100)
            if res is not None:
                return res
                
        return None

    start_time = time.time()
    deadline = 120 # Safety margin before 150s kill

    # Try ranks from 17 down to 1
    # We know 18 works. We want to find the minimal rank.
    
    # First, ensure schoolbook is valid (it is)
    best_result = (U_school, V_school, W_school)
    
    for R in range(17, 0, -1):
        if time.time() > deadline:
            break
            
        # Increase restarts for lower ranks as they are harder
        max_restarts = 20 if R < 15 else 10
        time_limit_per_rank = 10 if R < 15 else 5
        
        res = try_rank(R, U_school, V_school, W_school, max_restarts=max_restarts, time_limit=time_limit_per_rank)
        if res is not None:
            U, V, W = res
            best_result = (U, V, W)
            
    return {
        "U": best_result[0].tolist(),
        "V": best_result[1].tolist(),
        "W": best_result[2].tolist()
    }
