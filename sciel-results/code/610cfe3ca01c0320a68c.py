import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin

    best_rank = n * m * p
    best_decomp = None

    # Function to verify exactness
    def verify(U, V, W):
        # U, V, W are numpy arrays
        # Compute tensor
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        # Check if T_calc matches T
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def als_decompose(rank, max_iter=200, tol=1e-12, seed=None):
        """
        Alternating Least Squares for tensor decomposition.
        Returns (U, V, W) or None if it fails to converge or errors.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize randomly with small values to avoid divergence
        U = np.random.randn(U_size, rank) * 0.5
        V = np.random.randn(V_size, rank) * 0.5
        W = np.random.randn(W_size, rank) * 0.5
        
        prev_cost = np.inf
        
        for it in range(max_iter):
            # Update U
            # T (U, V, W) reshaped to (U, V*W)
            # M (V*W, rank) where M[(j,k), c] = V[j,c] * W[k,c]
            M = np.einsum('jc,kc->jkc', V, W).reshape(V_size * W_size, rank)
            # Solve M @ U_new.T = T_flat.T
            # T_flat.T is (V*W, U)
            try:
                U_new = np.linalg.lstsq(M, T.reshape(U_size, -1).T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None

            # Update V
            # For each v_idx (which is j*p+k), T[:, v_idx, :] is (U, W)
            # We want to find V[v_idx, :] such that V[v_idx, :] @ N.T approx T[:, v_idx, :].T
            # Where N (U*W, rank) has N[(u,w), c] = U[u,c] * W[w,c]
            # Actually, let's look at the structure:
            # T[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
            # Fix u, w. T[u, v, w] = V[v,c] * (U[u,c] W[w,c])
            # Let Q (U*W, rank) where Q[(u,w), c] = U[u,c] * W[w,c]
            # Then for a fixed v, the vector T[:, v, :].T (U*W, 1) = Q @ V[v, :].T (rank, 1)
            Q = np.einsum('uc,wc->uwc', U_new, W).reshape(U_size * W_size, rank)
            V_new = np.zeros((V_size, rank))
            for v in range(V_size):
                t_v = T[:, v, :].reshape(-1, 1) # (U*W, 1)
                try:
                    V_new[v] = np.linalg.lstsq(Q, t_v, rcond=None)[0].flatten()
                except np.linalg.LinAlgError:
                    return None

            # Update W
            # Similarly, fix u, v. T[u, v, w] = W[w,c] * (U[u,c] V[v,c])
            # Let R (U*V, rank) where R[(u,v), c] = U[u,c] * V[v,c]
            # Then for a fixed w, T[:, :, w].T (U*V, 1) = R @ W[w, :].T
            R = np.einsum('uc,vc->uvc', U_new, V_new).reshape(U_size * V_size, rank)
            W_new = np.zeros((W_size, rank))
            for w in range(W_size):
                t_w = T[:, :, w].reshape(-1, 1) # (U*V, 1)
                try:
                    W_new[w] = np.linalg.lstsq(R, t_w, rcond=None)[0].flatten()
                except np.linalg.LinAlgError:
                    return None

            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            # Calculate cost
            cost = 0.0
            for c in range(rank):
                T_calc = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                cost += np.sum((T_calc - T)**2)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
                
        # Final verification before returning
        # We don't snap here, just return the raw ALS result
        # The caller will snap and verify
        return U, V, W

    # Try ranks from 14 upwards
    # Theoretical lower bound is 14 (Strassen-like bounds for small matrices often suggest this range)
    # Schoolbook is 18.
    ranks_to_try = [14, 15, 16, 17]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Number of restarts depends on time left and rank
        # Rank 14 is the hardest, give it more tries
        max_restarts = 30 if rank == 14 else 15
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            seed = restart * 1000 + rank
            U, V, W = als_decompose(rank, max_iter=100, seed=seed)
            
            if U is None:
                continue

            # Snap and verify
            result = snap_and_verify(U, V, W)
            if result:
                U_final, V_final, W_final = result
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    # Fallback to schoolbook
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
