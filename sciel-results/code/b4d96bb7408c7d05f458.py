import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    # Build the tensor equation system
    # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Total equations: n*m*p = 36
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        # Check all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def polish(U, V, W, max_iter=50, tol=1e-10):
        """Polish a near-converged solution with tighter tolerance."""
        r = U.shape[1]
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Update U
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, r))
                    for k in range(p):
                        A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b_vec = np.ones(p)
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            # Update V
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, r))
                    for i in range(n):
                        A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b_vec = np.ones(n)
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            # Update W
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, r))
                    for j in range(m):
                        A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b_vec = np.ones(m)
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
        # Try rounding to half-integers
        for scale in [2, 4, 1]:  # half-integers, quarter-integers, integers
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify(U_r, V_r, W_r, 1e-9):
                return U_r, V_r, W_r
        return None
    
    # Try to find lower rank decompositions
    # Use ALS with multiple random initializations
    
    def try_rank(r, max_iter=300, tol=1e-10):
        if r >= best_rank:
            return None

        # Try multiple random initializations
        for init_seed in range(10):
            if time.time() > deadline:
                return None

            np.random.seed(42 + r * 1000 + init_seed * 137)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)

            # ALS iterations
            for it in range(max_iter):
                if time.time() > deadline:
                    return None

                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, r))
                        for k in range(p):
                            A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b_vec = np.ones(p)
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        except:
                            break
                    else:
                        continue
                    break
                else:
                    # Update V: for each (j,k), solve for V[j*p+k, :]
                    for j in range(m):
                        for k in range(p):
                            A_mat = np.zeros((n, r))
                            for i in range(n):
                                A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                            b_vec = np.ones(n)
                            try:
                                V[j * p + k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            except:
                                break
                        else:
                            continue
                        break
                    else:
                        # Update W: for each (k,i), solve for W[k*n+i, :]
                        for k in range(p):
                            for i in range(n):
                                A_mat = np.zeros((m, r))
                                for j in range(m):
                                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                                b_vec = np.ones(m)
                                try:
                                    W[k * n + i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                                except:
                                    break
                            else:
                                continue
                            break
                        else:
                            # Check convergence
                            if verify(U, V, W, tol):
                                polished = polish(U, V, W, max_iter=30, tol=1e-12)
                                if polished is not None:
                                    return polished
                                # Also try direct rounding
                                for scale in [2, 1]:
                                    U_r = np.round(U * scale) / scale
                                    V_r = np.round(V * scale) / scale
                                    W_r = np.round(W * scale) / scale
                                    if verify(U_r, V_r, W_r, 1e-9):
                                        return U_r, V_r, W_r
                                return None
                # End of iteration

            # Final check after all iterations
            if verify(U, V, W, tol):
                polished = polish(U, V, W, max_iter=30, tol=1e-12)
                if polished is not None:
                    return polished
                for scale in [2, 1]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    if verify(U_r, V_r, W_r, 1e-9):
                        return U_r, V_r, W_r

        return None
    
    # Try ranks from best_rank-1 down to some minimum
    # Focus on promising ranks
    candidate_ranks = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8]

    for r in candidate_ranks:
        if r >= best_rank:
            continue
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break  # Found the lowest rank we tried
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
