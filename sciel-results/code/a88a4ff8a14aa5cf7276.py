import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 9
    M = m * p  # 12
    P = p * n  # 12
    
    # Build the tensor T
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    best_rank = N * M * P
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute matricizations
    T_c = T.reshape(N * M, P)
    T_a = T.reshape(N, M * P)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)
    
    def check_solution(U, V, W, tol=1e-9):
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True, U_r, V_r, W_r
        return False, None, None, None

    def run_als(R, seed, max_iters=200):
        np.random.seed(seed)
        # Initialize with small values, potentially sparse
        U = np.random.randn(N, R) * 0.5
        V = np.random.randn(M, R) * 0.5
        W = np.random.randn(P, R) * 0.5
        
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                return None
            
            # Update W
            E_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(N * M, R)
            try:
                W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                W = W_T.T
            except:
                return None
            
            # Update U
            G_U = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(M * P, R)
            try:
                U_T, _, _, _ = np.linalg.lstsq(G_U.T, T_a.T, rcond=None)
                U = U_T.T
            except:
                return None
                
            # Update V
            G_V = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(N * P, R)
            try:
                V_T, _, _, _ = np.linalg.lstsq(G_V.T, T_b.T, rcond=None)
                V = V_T.T
            except:
                return None

            # Check convergence and validity
            E_W_new = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(N * M, R)
            T_calc_c = E_W_new @ W.T
            error = np.linalg.norm(T_calc_c - T_c)
            
            if error < 1e-7:
                valid, U_r, V_r, W_r = check_solution(U, V, W)
                if valid:
                    return U_r, V_r, W_r
        
        return None

    # Search ranks from 25 up to 35
    # We prioritize lower ranks.
    for R in range(25, 36):
        if time.time() - start_time > timeout:
            break
            
        # Try a few seeds for this rank
        # Use deterministic seeds to ensure reproducibility
        seeds = [R * 100 + i for i in range(10)]
        
        for seed in seeds:
            if time.time() - start_time > timeout:
                break
                
            res = run_als(R, seed)
            if res is not None:
                U_r, V_r, W_r = res
                best_rank = R
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break # Found a solution for this rank, move to next (lower) rank? 
                       # Wait, we are iterating 25 to 35. 
                       # If we find 25, that's the best. We should break out of the whole loop.
                # But what if 25 is impossible? We continue to 26.
                # So if we find a solution at R, we should check if we can go lower?
                # The loop goes 25, 26, 27...
                # If we find 25, we return immediately as it's the theoretical min.
                # If we find 26, we continue to check if 25 was missed? No, we tried 25 first.
                # So if we find a solution at R, and R is the smallest we've tried so far, it's the best.
                # Since we iterate 25, 26, 27..., the first valid solution found is the minimal rank found.
                # We should break the outer loop if we find a solution.
                break

        if best_rank < N * M * P:
            # We found a solution better than schoolbook. 
            # Since we iterate from low to high, this is the best found so far.
            # We can stop searching higher ranks.
            break

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
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
