# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from itertools import product

    start_time = time.time()
    DEATHLINE = 130.0
    n, m, p = 2, 5, 6

    def elapsed():
        return time.time() - start_time

    def build_trivial():
        """Build the trivial rank-60 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}, rank

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        rank = len(U[0])
        target_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_set.add((i * m + j, j * p + k, k * n + i))
        
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    s = 0.0
                    for r in range(rank):
                        s += U[u][r] * V[v][r] * W[w][r]
                    if (u, v, w) in target_set:
                        if abs(s - 1.0) > tol:
                            return False
                    else:
                        if abs(s) > tol:
                            return False
        return True

    def tensor_rank_als(R, max_iter=50, seed=42, tol=1e-10):
        """
        Try to find a rank-R decomposition using Alternating Least Squares.
        Returns (U, V, W) as lists or None if not found.
        """
        if elapsed() > DEATHLINE - 5:
            return None

        rng = np.random.RandomState(seed)
        
        # Initialize with random factors
        U = rng.randn(n * m, R)
        V = rng.randn(m * p, R)
        W = rng.randn(p * n, R)
        
        # Precompute the target tensor T
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[u_idx, v_idx, w_idx] = 1.0

        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None

        for it in range(max_iter):
            if elapsed() > DEATHLINE - 5:
                break
            
            # Update U
            M_VW = np.zeros((m * p, R))
            for r in range(R):
                M_VW[:, r] = V[:, r] * W[:, r]
            
            for u in range(n * m):
                try:
                    U[u, :], residuals, rank, sv = np.linalg.lstsq(M_VW.T, T[u, :, :].flatten(), rcond=None)
                except np.linalg.LinAlgError:
                    pass

            # Update V
            M_UW = np.zeros((n * m, R))
            for r in range(R):
                M_UW[:, r] = U[:, r] * W[:, r]
            
            for v in range(m * p):
                try:
                    V[v, :], residuals, rank, sv = np.linalg.lstsq(M_UW.T, T[:, v, :].flatten(), rcond=None)
                except np.linalg.LinAlgError:
                    pass

            # Update W
            M_UV = np.zeros((n * m * m * p, R))
            idx = 0
            for u in range(n * m):
                for v in range(m * p):
                    M_UV[idx, :] = U[u, :] * V[v, :]
                    idx += 1
            
            for w in range(p * n):
                try:
                    W[w, :], residuals, rank, sv = np.linalg.lstsq(M_UV.T, T[:, :, w].flatten(), rcond=None)
                except np.linalg.LinAlgError:
                    pass

            # Calculate loss
            loss = 0.0
            for u in range(n * m):
                for v in range(m * p):
                    vec = np.dot(U[u, :] * V[v, :], W.T)
                    loss += np.sum((vec - T[u, v, :]) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < tol:
                break

        if best_loss < 1e-8:
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        mat[i][j] = round(mat[i][j] * 2) / 2.0
            
            if verify_decomposition(U_list, V_list, W_list, tol=1e-8):
                return U_list, V_list, W_list
        
        return None

    # Start with trivial decomposition
    best_result, best_rank = build_trivial()
    
    if elapsed() > DEATHLINE - 5:
        return best_result
    
    # Try to find lower-rank decompositions
    found_lower = False
    for R in range(40, 60):
        if elapsed() > DEATHLINE - 10:
            break
        
        result = tensor_rank_als(R, max_iter=50, seed=R, tol=1e-10)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = R
            found_lower = True
            break
    
    if found_lower:
        for R in range(30, best_rank):
            if elapsed() > DEATHLINE - 10:
                break
            
            result = tensor_rank_als(R, max_iter=50, seed=R, tol=1e-10)
            if result is not None:
                U_list, V_list, W_list = result
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = R
                break
    
    return best_result
# EVOLVE-BLOCK-END
