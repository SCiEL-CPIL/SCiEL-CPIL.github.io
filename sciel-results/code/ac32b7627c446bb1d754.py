import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    def build_strassen_decomposition():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_term(u_indices, v_indices, w_indices, coeffs=None):
            nonlocal col
            if coeffs is None:
                coeffs = [1] * len(u_indices)
            for idx, c in zip(u_indices, coeffs):
                U[idx][col] += c
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

    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_rank_reduction(U, V, W, target_rank, max_iters=5000):
        """Try to reduce rank using local search."""
        n, m, p = DIMENSIONS
        current_rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        best_rank = current_rank
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        
        random.seed(123)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            if current_rank <= target_rank:
                break
            
            # Randomly perturb one factor
            factor = random.choice(['U', 'V', 'W'])
            row = random.randint(0, (n*m if factor=='U' else m*p if factor=='V' else p*n) - 1)
            col = random.randint(0, current_rank - 1)
            
            old_val = (U_arr if factor=='U' else V_arr if factor=='V' else W_arr)[row, col]
            new_val = old_val + random.gauss(0, 0.1)
            
            arr = U_arr if factor=='U' else V_arr if factor=='V' else W_arr
            arr[row, col] = new_val
            
            # Check if decomposition still holds
            if verify_decomposition(arr.tolist(), 
                                    V_arr.tolist() if factor=='U' else U_arr.tolist(),
                                    W_arr.tolist() if factor=='V' else U_arr.tolist(),
                                    n, m, p, tol=1e-6):
                # Try to see if we can zero out a column
                for c in range(current_rank):
                    if factor == 'U':
                        if np.all(np.abs(U_arr[:, c]) < 1e-8):
                            U_arr = np.delete(U_arr, c, axis=1)
                            V_arr = np.delete(V_arr, c, axis=1)
                            W_arr = np.delete(W_arr, c, axis=1)
                            current_rank -= 1
                            best_rank = min(best_rank, current_rank)
                            best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
                            break
                    elif factor == 'V':
                        if np.all(np.abs(V_arr[:, c]) < 1e-8):
                            U_arr = np.delete(U_arr, c, axis=1)
                            V_arr = np.delete(V_arr, c, axis=1)
                            W_arr = np.delete(W_arr, c, axis=1)
                            current_rank -= 1
                            best_rank = min(best_rank, current_rank)
                            best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
                            break
                    else:
                        if np.all(np.abs(W_arr[:, c]) < 1e-8):
                            U_arr = np.delete(U_arr, c, axis=1)
                            V_arr = np.delete(V_arr, c, axis=1)
                            W_arr = np.delete(W_arr, c, axis=1)
                            current_rank -= 1
                            best_rank = min(best_rank, current_rank)
                            best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
                            break
            else:
                arr[row, col] = old_val
        
        if best_rank < current_rank:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        return None

    base_solution = build_strassen_decomposition()
    
    if not verify_decomposition(base_solution["U"], base_solution["V"], base_solution["W"], n, m, p):
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
    
    # Try to reduce rank
    reduced = try_rank_reduction(base_solution["U"], base_solution["V"], base_solution["W"], 38)
    
    if reduced is not None:
        return reduced
    
    return base_solution

# EVOLVE-BLOCK-END
