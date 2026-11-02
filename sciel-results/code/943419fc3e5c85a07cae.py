import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    # Shapes
    # U: (n*m, R) = (4, R)
    # V: (m*p, R) = (8, R)
    # W: (p*n, R) = (8, R)
    
    # Target tensor entries T[i, j, j', k] = delta[j, j']
    # Equation: sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r] = delta[j, j']
    
    # We will try to find a rank 13 solution.
    # If we fail, we fall back to rank 14.
    
    start_time = time.time()
    max_time = 100.0 # Safety margin
    
    best_result = None
    best_error = float('inf')
    
    # Helper to check error
    def check_error(U, V, W):
        R = U.shape[1]
        err = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        # Calculate LHS
                        val = 0.0
                        for r in range(R):
                            val += W[k*n + i, r] * U[i*m + j, r] * V[jp*p + k, r]
                        target = 1.0 if j == jp else 0.0
                        err += (val - target) ** 2
        return err

    # Helper to solve for one matrix given the other two
    # We rewrite the equation as A x = b
    # For W:
    # sum_r W[idx_w, r] (U[idx_u, r] V[idx_v, r]) = target
    # Let C_r = U[idx_u, r] V[idx_v, r]
    # This is not a simple linear system in W because the RHS depends on which i,k,j,jp we are looking at.
    # Actually, for a fixed (i, k, j, jp), the equation is linear in W[k*n+i, :].
    # But we have many equations.
    # We can form a large linear system.
    
    # Let's vectorize the variables.
    # For W: Variables are W[0..7, 0..R-1].
    # Equation for (i, k, j, jp):
    # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[jp*p+k, r]) = delta[j, jp]
    # This is a linear equation in the vector W[k*n+i, :].
    # Let w_vec = W[k*n+i, :].
    # coeff_vec[r] = U[i*m+j, r] * V[jp*p+k, r].
    # dot(w_vec, coeff_vec) = target.
    
    # We have 32 such equations.
    # We can group them by the index of W.
    # For each w_idx in 0..7:
    #   Collect all equations involving W[w_idx, :].
    #   Solve least squares for W[w_idx, :].
    
    def solve_for_W(U, V):
        R = U.shape[1]
        W = np.zeros((p*n, R))
        for w_idx in range(p*n):
            k = w_idx // n
            i = w_idx % n
            A_sys = []
            b_sys = []
            for j in range(m):
                for jp in range(m):
                    target = 1.0 if j == jp else 0.0
                    coeff = np.array([U[i*m + j, r] * V[jp*p + k, r] for r in range(R)])
                    A_sys.append(coeff)
                    b_sys.append(target)
            A_sys = np.array(A_sys)
            b_sys = np.array(b_sys)
            # Solve A_sys.T @ W[w_idx] = b_sys
            # Using least squares
            sol, residuals, rank, s = np.linalg.lstsq(A_sys.T, b_sys, rcond=None)
            W[w_idx, :] = sol
        return W

    def solve_for_U(V, W):
        R = V.shape[1]
        U = np.zeros((n*m, R))
        for u_idx in range(n*m):
            i = u_idx // m
            j = u_idx % m
            A_sys = []
            b_sys = []
            for k in range(p):
                for jp in range(m):
                    target = 1.0 if j == jp else 0.0
                    coeff = np.array([W[k*n + i, r] * V[jp*p + k, r] for r in range(R)])
                    A_sys.append(coeff)
                    b_sys.append(target)
            A_sys = np.array(A_sys)
            b_sys = np.array(b_sys)
            sol, residuals, rank, s = np.linalg.lstsq(A_sys.T, b_sys, rcond=None)
            U[u_idx, :] = sol
        return U

    def solve_for_V(U, W):
        R = U.shape[1]
        V = np.zeros((m*p, R))
        for v_idx in range(m*p):
            jp = v_idx // p
            k = v_idx % p
            A_sys = []
            b_sys = []
            for i in range(n):
                for j in range(m):
                    target = 1.0 if j == jp else 0.0
                    coeff = np.array([W[k*n + i, r] * U[i*m + j, r] for r in range(R)])
                    A_sys.append(coeff)
                    b_sys.append(target)
            A_sys = np.array(A_sys)
            b_sys = np.array(b_sys)
            sol, residuals, rank, s = np.linalg.lstsq(A_sys.T, b_sys, rcond=None)
            V[v_idx, :] = sol
        return V

    def try_rank(R, seed):
        np.random.seed(seed)
        # Initialize
        U = np.random.rand(n*m, R)
        V = np.random.rand(m*p, R)
        W = np.random.rand(p*n, R)
        
        for it in range(100):
            if time.time() - start_time > max_time:
                return None, float('inf')
            
            W = solve_for_W(U, V)
            U = solve_for_U(V, W)
            V = solve_for_V(U, W)
            
            err = check_error(U, V, W)
            if err < 1e-10:
                return U, V, W, err
        
        return U, V, W, check_error(U, V, W)

    # Try rank 13
    for seed in range(50):
        if time.time() - start_time > max_time:
            break
        res, err = try_rank(13, seed)
        if res is not None and err < 1e-10:
            # Found a rank 13 solution
            # Convert to list of lists
            U_list = res[0].tolist()
            V_list = res[1].tolist()
            W_list = res[2].tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        if err < best_error:
            best_error = err
            best_result = (res[0], res[1], res[2])
            
    # If rank 13 not found, try rank 14
    for seed in range(20):
        if time.time() - start_time > max_time:
            break
        res, err = try_rank(14, seed)
        if res is not None and err < 1e-10:
            U_list = res[0].tolist()
            V_list = res[1].tolist()
            W_list = res[2].tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        if err < best_error:
            best_error = err
            best_result = (res[0], res[1], res[2])

    # Fallback to a known rank 14 Strassen-like solution if numerical search fails or is too slow
    # We use the hardcoded solution from the prompt as a safe fallback
    rank = 14
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    for r_local in range(7):
        r_global = r_local
        u_terms, v_terms, w_terms = strassen_terms[r_local]
        for idx, val in u_terms:
            U[idx][r_global] = val
        for idx, val in v_terms:
            j = idx // 2
            k = idx % 2
            global_v_idx = j * 4 + k
            V[global_v_idx][r_global] = val
        for idx, val in w_terms:
            k = idx // 2
            i = idx % 2
            global_w_idx = k * 2 + i
            W[global_w_idx][r_global] = val

    for r_local in range(7):
        r_global = 7 + r_local
        u_terms, v_terms, w_terms = strassen_terms[r_local]
        for idx, val in u_terms:
            U[idx][r_global] = val
        for idx, val in v_terms:
            j = idx // 2
            k_local = idx % 2
            k_global = k_local + 2
            global_v_idx = j * 4 + k_global
            V[global_v_idx][r_global] = val
        for idx, val in w_terms:
            k_local = idx // 2
            i = idx % 2
            k_global = k_local + 2
            global_w_idx = k_global * 2 + i
            W[global_w_idx][r_global] = val

    return {"U": U, "V": V, "W": W}
