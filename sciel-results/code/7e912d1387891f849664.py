# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin

    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    np.random.seed(42)

    def try_rank(r, max_iter=50, restarts=10):
        """
        Attempt to find a rank-r decomposition using ALS.
        Returns a dict if successful, None otherwise.
        """
        for _ in range(restarts):
            if time.time() > deadline - 5:
                return None

            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                for u_idx in range(n*m):
                    i = u_idx // m
                    j = u_idx % m
                    
                    A_u = np.zeros((m*p * p*n, r))
                    b_u = np.zeros(m*p * p*n)
                    
                    for v_idx in range(m*p):
                        for w_idx in range(p*n):
                            row_idx = v_idx * (p*n) + w_idx
                            
                            j_v = v_idx // p
                            k_v = v_idx % p
                            i_w = w_idx % n
                            k_w = w_idx // n
                            
                            if i_w == i and j_v == j and k_v == k_w:
                                b_u[row_idx] = 1.0
                            else:
                                b_u[row_idx] = 0.0
                            
                            for t in range(r):
                                A_u[row_idx, t] = V[v_idx, t] * W[w_idx, t]
                    
                    sol, residuals, rank_u, s = np.linalg.lstsq(A_u, b_u, rcond=None)
                    U[u_idx, :] = sol

                # Update V
                for v_idx in range(m*p):
                    j = v_idx // p
                    k = v_idx % p
                    
                    A_v = np.zeros((n*m * p*n, r))
                    b_v = np.zeros(n*m * p*n)
                    
                    for u_idx in range(n*m):
                        for w_idx in range(p*n):
                            row_idx = u_idx * (p*n) + w_idx
                            
                            i_u = u_idx // m
                            j_u = u_idx % m
                            i_w = w_idx % n
                            k_w = w_idx // n
                            
                            if i_u == i_w and j_u == j and k_w == k:
                                b_v[row_idx] = 1.0
                            else:
                                b_v[row_idx] = 0.0
                            
                            for t in range(r):
                                A_v[row_idx, t] = U[u_idx, t] * W[w_idx, t]
                    
                    sol, residuals, rank_v, s = np.linalg.lstsq(A_v, b_v, rcond=None)
                    V[v_idx, :] = sol

                # Update W
                for w_idx in range(p*n):
                    i = w_idx % n
                    k = w_idx // n
                    
                    A_w = np.zeros((n*m * m*p, r))
                    b_w = np.zeros(n*m * m*p)
                    
                    for u_idx in range(n*m):
                        for v_idx in range(m*p):
                            row_idx = u_idx * (m*p) + v_idx
                            
                            i_u = u_idx // m
                            j_u = u_idx % m
                            j_v = v_idx // p
                            k_v = v_idx % p
                            
                            if i_u == i and j_u == j_v and k_v == k:
                                b_w[row_idx] = 1.0
                            else:
                                b_w[row_idx] = 0.0
                            
                            for t in range(r):
                                A_w[row_idx, t] = U[u_idx, t] * V[v_idx, t]
                    
                    sol, residuals, rank_w, s = np.linalg.lstsq(A_w, b_w, rcond=None)
                    W[w_idx, :] = sol
                
                # Check convergence
                max_err = 0
                for u_idx in range(n*m):
                    for v_idx in range(m*p):
                        for w_idx in range(p*n):
                            val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                            i_u = u_idx // m
                            j_u = u_idx % m
                            j_v = v_idx // p
                            k_v = v_idx % p
                            i_w = w_idx % n
                            k_w = w_idx // n
                            expected = 1.0 if (i_u == i_w and j_u == j_v and k_v == k_w) else 0.0
                            err = abs(val - expected)
                            if err > max_err:
                                max_err = err
                            
                if max_err < 1e-8:
                    # Try rounding to half-integers
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    max_err_rounded = 0
                    for u_idx in range(n*m):
                        for v_idx in range(m*p):
                            for w_idx in range(p*n):
                                val = np.dot(U_int[u_idx, :], np.multiply(V_int[v_idx, :], W_int[w_idx, :]))
                                i_u = u_idx // m
                                j_u = u_idx % m
                                j_v = v_idx // p
                                k_v = v_idx % p
                                i_w = w_idx % n
                                k_w = w_idx // n
                                expected = 1.0 if (i_u == i_w and j_u == j_v and k_v == k_w) else 0.0
                                err = abs(val - expected)
                                if err > max_err_rounded:
                                    max_err_rounded = err
                    
                    if max_err_rounded < 1e-8:
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                    else:
                        # Try integer rounding?
                        U_int2 = np.round(U)
                        V_int2 = np.round(V)
                        W_int2 = np.round(W)
                        
                        max_err_int = 0
                        for u_idx in range(n*m):
                            for v_idx in range(m*p):
                                for w_idx in range(p*n):
                                    val = np.dot(U_int2[u_idx, :], np.multiply(V_int2[v_idx, :], W_int2[w_idx, :]))
                                    i_u = u_idx // m
                                    j_u = u_idx % m
                                    j_v = v_idx // p
                                    k_v = v_idx % p
                                    i_w = w_idx % n
                                    k_w = w_idx // n
                                    expected = 1.0 if (i_u == i_w and j_u == j_v and k_v == k_w) else 0.0
                                    err = abs(val - expected)
                                    if err > max_err_int:
                                        max_err_int = err
                                        
                        if max_err_int < 1e-8:
                            return {
                                "U": U_int2.tolist(),
                                "V": V_int2.tolist(),
                                "W": W_int2.tolist()
                            }
                        else:
                            # Rounding failed, continue searching or return None
                            return None

            return None

    # Try ranks
    lb = max(n*m, m*p, p*n)
    # Start from a reasonable guess. 
    for r in range(55, lb - 1, -1):
        if time.time() > deadline - 10:
            break
            
        result = try_rank(r, max_iter=50, restarts=5)
        if result:
            return result

    return best
# EVOLVE-BLOCK-END
