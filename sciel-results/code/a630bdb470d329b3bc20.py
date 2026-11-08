import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Fallback rank 39 construction
    def construct_39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx in u_indices:
                U[idx][col] += 1
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1

        def add_strassen_block(k_start):
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1
            
            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1
            
            add_term([0, 4], [b00, b11], [c00, c11])
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1
            
            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1
            
            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1
            
            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1
            
            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1
            
            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1

        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])
        
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])
            
        return {"U": U, "V": V, "W": W}

    # Try to find a better rank using search
    start_time = time.time()
    deadline = start_time + 100 # 20s margin
    
    # Prepare triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                triples.append((alpha, beta, gamma))
    
    num_eq = len(triples)
    
    # Try ranks 38 down to 35
    for R in range(38, 34, -1):
        if time.time() > deadline:
            break
        num_vars = R * (n * m + m * p + p * n)
        
        def residuals(params):
            U = params[0 : R * n * m].reshape(R, n * m).T
            V = params[R * n * m : R * (n * m + m * p)].reshape(R, m * p).T
            W = params[R * (n * m + m * p) :].reshape(R, p * n).T
            
            res = np.zeros(num_eq)
            for idx, (a, b, g) in enumerate(triples):
                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                res[idx] = val - 1.0
            return res
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            params0 = np.random.randn(num_vars)
            params0 /= np.sqrt(num_vars)
            
            try:
                sol = least_squares(residuals, params0, method='lm', max_nfev=500)
                if sol.success and sol.cost < 1e-8:
                    U = sol.x[0 : R * n * m].reshape(R, n * m).T
                    V = sol.x[R * n * m : R * (n * m + m * p)].reshape(R, m * p).T
                    W = sol.x[R * (n * m + m * p) :].reshape(R, p * n).T
                    
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    valid = True
                    for idx, (a, b, g) in enumerate(triples):
                        val = np.sum(U_int[a, :] * V_int[b, :] * W_int[g, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if valid:
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
            except Exception:
                pass
    
    # Fallback
    return construct_39()
# EVOLVE-BLOCK-END
