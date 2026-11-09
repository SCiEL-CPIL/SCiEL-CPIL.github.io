import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Precompute the target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += U_arr[:, r, None, None] * V_arr[:, r][None, :, None] * W_arr[:, r][None, None, :]
        
        return np.allclose(T, target, atol=1e-10)

    def tensor_error(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U)
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += U[:, r, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        return np.sum((T - target) ** 2)

    def als_solve(r, max_iters=100, seed=0):
        """Alternating least squares for rank-r decomposition."""
        n, m, p = DIMENSIONS
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Update U
            M = np.zeros((r, m*p, p*n))
            for r_idx in range(r):
                M[r_idx] = np.outer(V[:, r_idx], W[:, r_idx])
            M_flat = M.reshape(r, -1).T
            
            for a in range(n*m):
                U[a] = np.linalg.lstsq(M_flat, target[a].reshape(-1), rcond=None)[0]
            
            # Update V
            N = np.zeros((r, n*m, p*n))
            for r_idx in range(r):
                N[r_idx] = np.outer(U[:, r_idx], W[:, r_idx])
            N_flat = N.reshape(r, -1).T
            
            for b in range(m*p):
                V[b] = np.linalg.lstsq(N_flat, target[:, b, :].reshape(-1), rcond=None)[0]
            
            # Update W
            P = np.zeros((r, n*m, m*p))
            for r_idx in range(r):
                P[r_idx] = np.outer(U[:, r_idx], V[:, r_idx])
            P_flat = P.reshape(r, -1).T
            
            for c in range(p*n):
                W[c] = np.linalg.lstsq(P_flat, target[:, :, c].reshape(-1), rcond=None)[0]
            
            err = tensor_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-12:
                break
        
        return best_U, best_V, best_W, best_err

    def try_rank(r, n_restarts=10):
        """Try to find a rank-r decomposition."""
        for restart in range(n_restarts):
            U, V, W, err = als_solve(r, max_iters=50, seed=100 + restart)
            
            if err < 1e-10:
                # Round to half-integers
                U_r = [[round(x*2)/2 for x in row] for row in U.tolist()]
                V_r = [[round(x*2)/2 for x in row] for row in V.tolist()]
                W_r = [[round(x*2)/2 for x in row] for row in W.tolist()]
                
                if verify(U_r, V_r, W_r):
                    return {"U": U_r, "V": V_r, "W": W_r}
        
        return None

    start_rank = max(n*m, m*p, p*n)  # 25
    deadline = time.time() + 120
    
    for r in range(start_rank, best_rank):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r, n_restarts=5)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
