import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s for cleanup and safety margin

    u_rows = n * m
    v_rows = m * p
    w_rows = p * n

    # Build the multiplication tensor T
    # T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i for some i,j,k
    T = np.zeros((u_rows, v_rows, w_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def check_exact(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    def discrete_local_search(U, V, W, max_iters=1000):
        """
        Local search in the discrete space of half-integers.
        Tries to reduce the error by adjusting individual entries.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Ensure we start with half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        current_err = np.max(np.abs(np.einsum('uc,vc,wc->uvw', U, V, W) - T))
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            if current_err < 1e-9:
                break
                
            improved = False
            # Try adjusting U
            for i in range(U.shape[0]):
                for j in range(U.shape[1]):
                    old_val = U[i, j]
                    # Try neighbors: old_val +/- 0.5
                    for delta in [-0.5, 0.5]:
                        new_val = old_val + delta
                        U[i, j] = new_val
                        err = np.max(np.abs(np.einsum('uc,vc,wc->uvw', U, V, W) - T))
                        if err < current_err:
                            current_err = err
                            improved = True
                        else:
                            U[i, j] = old_val
                            
            if not improved:
                # Try adjusting V
                for i in range(V.shape[0]):
                    for j in range(V.shape[1]):
                        old_val = V[i, j]
                        for delta in [-0.5, 0.5]:
                            new_val = old_val + delta
                            V[i, j] = new_val
                            err = np.max(np.abs(np.einsum('uc,vc,wc->uvw', U, V, W) - T))
                            if err < current_err:
                                current_err = err
                                improved = True
                            else:
                                V[i, j] = old_val
                                
            if not improved:
                # Try adjusting W
                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        old_val = W[i, j]
                        for delta in [-0.5, 0.5]:
                            new_val = old_val + delta
                            W[i, j] = new_val
                            err = np.max(np.abs(np.einsum('uc,vc,wc->uvw', U, V, W) - T))
                            if err < current_err:
                                current_err = err
                                improved = True
                            else:
                                W[i, j] = old_val
                                
            if not improved:
                # Try larger jumps if stuck? Or restart?
                # For now, break if no single step improvement
                break
                
        if current_err < 1e-9:
            return U, V, W
        return None

    def try_rank(r, seed=42, max_iter=200):
        """Alternating minimization to find rank-r decomposition."""
        if time.time() > deadline - 5:
            return None

        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(u_rows, r) * 0.1
        V = np.random.randn(v_rows, r) * 0.1
        W = np.random.randn(w_rows, r) * 0.1

        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None

            # Update U: fix V, W
            TW = V[:, None, :] * W[None, :, :]  # (v_rows, w_rows, r)
            h_U = np.einsum('abc,bcd->ad', T, TW)  # (u_rows, r)
            G_U = np.einsum('bcd,bce->de', TW, TW)  # (r, r)
            try:
                U = np.linalg.solve(G_U, h_U.T).T
            except np.linalg.LinAlgError:
                pass

            # Update V: fix U, W
            UW = U[:, None, :] * W[None, :, :]  # (u_rows, w_rows, r)
            h_V = np.einsum('abc,acd->bd', T, UW)  # (v_rows, r)
            G_V = np.einsum('acd,ace->de', UW, UW)  # (r, r)
            try:
                V = np.linalg.solve(G_V, h_V.T).T
            except np.linalg.LinAlgError:
                pass

            # Update W: fix U, V
            UV = U[:, None, :] * V[None, :, :]  # (u_rows, v_rows, r)
            h_W = np.einsum('abc,abd->cd', T, UV)  # (w_rows, r)
            G_W = np.einsum('abd,abe->de', UV, UV)  # (r, r)
            try:
                W = np.linalg.solve(G_W, h_W.T).T
            except np.linalg.LinAlgError:
                pass

            # Early exit check
            if it % 10 == 0:
                T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
                if np.allclose(T_recon, T, atol=1e-8):
                    break

        # Try rounding to half-integers and then discrete local search
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if already exact
        if check_exact(U_r, V_r, W_r):
            return (r, U_r, V_r, W_r)

        # Try discrete local search to fix the error
        result = discrete_local_search(U_r, V_r, W_r)
        if result is not None:
            return (r, result[0], result[1], result[2])

        return None

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(u_rows)]
        V = [[0] * rank for _ in range(v_rows)]
        W = [[0] * rank for _ in range(w_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Try ranks from a reasonable lower bound up to schoolbook - 1
    # Lower bound: max(n*p, n*m, m*p) = max(15, 9, 15) = 15
    # But for 3x3x5, known results suggest rank might be around 14-16
    ranks_to_try = list(range(15, best_rank))

    seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456, 789, 1000, 2024, 3141, 2718]

    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break

        for seed in seeds:
            if time.time() > deadline - 5:
                break
            result = try_rank(r, seed)
            if result is not None:
                r_found, U_f, V_f, W_f = result
                if r_found < best_rank:
                    best_rank = r_found
                    best_result = {
                        "U": U_f.tolist(),
                        "V": V_f.tolist(),
                        "W": W_f.tolist()
                    }
                    break  # Found this rank, move to next lower rank

    return best_result
