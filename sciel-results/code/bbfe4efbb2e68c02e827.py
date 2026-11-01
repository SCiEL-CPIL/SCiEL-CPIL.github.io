import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 2, 12
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        T_calc = np.zeros((n, m, p))
        for s in range(rank):
            u_col = np.array(U)[:, s].reshape(n, m)
            v_col = np.array(V)[:, s].reshape(m, p)
            w_col = np.array(W)[:, s].reshape(p, n)
            # T[i,j,k] += U[i,j,s] * V[j,k,s] * W[k,i,s]
            # Note: W index is k*n + i, so w_col[k, i] corresponds to W[k*n+i, s]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_calc[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        
        T_target = np.ones((n, m, p))
        max_err = np.max(np.abs(T_calc - T_target))
        return max_err < 1e-9

    def search_rank(R, n_sub, m_sub, p_sub, num_restarts=50, maxiter=2000):
        """
        Search for a rank-R decomposition of the n_sub x m_sub x p_sub multiplication tensor.
        Returns (U, V, W) as lists of lists if found, else None.
        """
        if time.time() > deadline:
            return None

        dim_U = n_sub * m_sub
        dim_V = m_sub * p_sub
        dim_W = p_sub * n_sub
        
        total_vars = dim_U * R + dim_V * R + dim_W * R

        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = np.reshape(params[:dim_U*R], (dim_U, R))
            V = np.reshape(params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(params[dim_U*R+dim_V*R:], (dim_W, R))
            
            T_calc = np.zeros((n_sub, m_sub, p_sub))
            for s in range(R):
                u_col = U[:, s].reshape(n_sub, m_sub)
                v_col = V[:, s].reshape(m_sub, p_sub)
                w_col = W[:, s].reshape(p_sub, n_sub)
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            T_calc[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
            
            T_target = np.ones((n_sub, m_sub, p_sub))
            error = np.sum((T_calc - T_target)**2)
            return error

        best_error = float('inf')
        best_params = None
        
        np.random.seed(42)
        for _ in range(num_restarts):
            if time.time() > deadline - 5:
                break
            # Initialize with small random values
            params = np.random.uniform(-1, 1, size=total_vars)
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': maxiter, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
            if best_error < 1e-8:
                break

        if best_error < 1e-6 and best_params is not None:
            U = np.reshape(best_params[:dim_U*R], (dim_U, R))
            V = np.reshape(best_params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(best_params[dim_U*R+dim_V*R:], (dim_W, R))
            
            # Snap to half-integers
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            # Verify the snapped solution
            if verify_decomposition(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n_sub, m_sub, p_sub):
                return U_snapped, V_snapped, W_snapped
        return None

    def assemble_full(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, p_full, n, m):
        """
        Assemble the full 2x2x12 decomposition from a block decomposition.
        """
        num_blocks = p_full // p_sub
        rank_sub = len(U_sub[0])
        full_rank = num_blocks * rank_sub
        
        full_U = [[0] * full_rank for _ in range(n * m)]
        full_V = [[0] * full_rank for _ in range(m * p_full)]
        full_W = [[0] * full_rank for _ in range(p_full * n)]
        
        for block in range(num_blocks):
            offset = block * rank_sub
            for s in range(rank_sub):
                r = offset + s
                # U is same for all blocks (A is 2x2)
                for i in range(n_sub):
                    for j in range(m_sub):
                        full_U[i*m_sub+j][r] = float(U_sub[i*m_sub+j, s])
                
                # V maps to specific columns in B
                for j in range(m_sub):
                    for k in range(p_sub):
                        k_glob = block * p_sub + k
                        full_V[j*p_full + k_glob][r] = float(V_sub[j*p_sub+k, s])
                        
                # W maps to specific columns in C
                for k in range(p_sub):
                    for i in range(n_sub):
                        k_glob = block * p_sub + k
                        full_W[k_glob*n + i][r] = float(W_sub[k*n_sub+i, s])
                        
        return {"U": full_U, "V": full_V, "W": full_W}

    # Strategy 1: Rank 10 for 2x2x4 (Total Rank 30)
    # Try to find a rank-10 decomposition for 2x2x4
    res = search_rank(10, 2, 2, 4, num_restarts=30, maxiter=1000)
    if res:
        U_sub, V_sub, W_sub = res
        return assemble_full(U_sub, V_sub, W_sub, 2, 2, 4, 12, 2, 2)

    # Strategy 2: Rank 8 for 2x2x3 (Total Rank 32)
    # Try to find a rank-8 decomposition for 2x2x3
    res = search_rank(8, 2, 2, 3, num_restarts=30, maxiter=1000)
    if res:
        U_sub, V_sub, W_sub = res
        return assemble_full(U_sub, V_sub, W_sub, 2, 2, 3, 12, 2, 2)

    # Fallback: Rank 42 Strassen decomposition
    def get_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Standard Strassen coefficients for 2x2x2
        strassen_coeffs = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        r = 0
        for b in range(6):
            for s in range(7):
                u, v, w = strassen_coeffs[s]
                for i_loc in range(2):
                    for j_loc in range(2):
                        val = u[i_loc * 2 + j_loc]
                        if val != 0:
                            U[i_loc * m + j_loc][r] = val
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = v[j_loc * 2 + k_loc]
                        if val != 0:
                            V[j_loc * p + (2 * b + k_loc)][r] = val
                for k_loc in range(2):
                    for i_loc in range(2):
                        val = w[k_loc * 2 + i_loc]
                        if val != 0:
                            W[(2 * b + k_loc) * n + i_loc][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    return get_strassen_42()
