"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    def tensor_residual(U, V, W):
        rank = U.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    total += (s - 1.0) ** 2
        return total
    
    def is_half_integer(arr, tol=1e-9):
        flat = arr.flatten()
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)
    
    def round_to_half(arr):
        return np.round(arr * 2) / 2
    
    def verify_solution(U, V, W, tol=1e-6):
        if not (is_half_integer(U, 1e-8) and is_half_integer(V, 1e-8) and is_half_integer(W, 1e-8)):
            return False
        res = tensor_residual(U, V, W)
        return res < tol
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def try_rank_14():
        """
        Try to construct rank-14 decomposition.
        Split: A (2x3) * B (3x3) = C (2x3)
        Split B's rows: B = [B1 (2x3); B2 (1x3)]
        C = A1*B1 + A2*B2 where A1 is 2x2 (first 2 cols of A), A2 is 2x1 (last col of A)
        A1*B1: 2x2 * 2x3, rank <= 11
        A2*B2: 2x1 * 1x3, rank = 3
        Total: 14
        """
        target_rank = 14
        U = np.zeros((n*m, target_rank))  # (6, 14)
        V = np.zeros((m*p, target_rank))  # (9, 14)
        W = np.zeros((p*n, target_rank))  # (6, 14)
        
        # Part 1: A1 (2x2) * B1 (2x3) -> C (2x3)
        # i in {0,1}, j in {0,1}, k in {0,1,2}
        # 11 terms: indices 0-10
        # Start with 12 schoolbook terms, use 11 by merging
        entries = [(i, j, k) for i in range(2) for j in range(2) for k in range(3)]
        for idx, (i, j, k) in enumerate(entries):
            if idx < 11:
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
        
        # Part 2: A2 (2x1) * B2 (1x3) -> C (2x3)
        # i in {0,1}, j=2, k in {0,1,2}
        # 3 terms: indices 11-13
        for k in range(3):
            r = 11 + k
            U[0*m + 2, r] = 1.0  # i=0, j=2
            U[1*m + 2, r] = 1.0  # i=1, j=2
            V[2*p + k, r] = 1.0  # j=2, k=k
            W[k*n + 0, r] = 1.0  # k, i=0
            W[k*n + 1, r] = 1.0  # k, i=1
        
        return U, V, W
    
    # Try rank 14
    if time.time() < deadline - 60:
        U14, V14, W14 = try_rank_14()
        res = tensor_residual(U14, V14, W14)
        if res < 1e-6:
            if verify_solution(U14, V14, W14, 1e-6):
                best = {"U": U14.tolist(), "V": V14.tolist(), "W": W14.tolist()}
                best_rank = 14
                return best
    
    # Optimize from rank-14 start
    if time.time() < deadline - 30:
        U14, V14, W14 = try_rank_14()
        x0 = np.concatenate([U14.flatten(), V14.flatten(), W14.flatten()])
        
        def obj(x):
            nr = 14
            U_flat = x[:6*nr].reshape(6, nr)
            V_flat = x[6*nr : 6*nr + 9*nr].reshape(9, nr)
            W_flat = x[6*nr + 9*nr :].reshape(6, nr)
            return tensor_residual(U_flat, V_flat, W_flat)
        
        from scipy.optimize import minimize
        res_opt = minimize(obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if res_opt.fun < 1e-8:
            U_sol = res_opt.x[:6*14].reshape(6, 14)
            V_sol = res_opt.x[6*14 : 6*14 + 9*14].reshape(9, 14)
            W_sol = res_opt.x[6*14 + 9*14 :].reshape(6, 14)
            
            if verify_solution(U_sol, V_sol, W_sol, 1e-6):
                best = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                best_rank = 14
                return best
            else:
                U_r = round_to_half(U_sol)
                V_r = round_to_half(V_sol)
                W_r = round_to_half(W_sol)
                if verify_solution(U_r, V_r, W_r, 1e-6):
                    best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = 14
                    return best
    
    # Fallback: try ranks 15-17
    for target_rank in range(15, rank_sb):
        if time.time() > deadline - 15:
            break
        
        for attempt in range(8):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + target_rank * 100 + attempt)
            U = np.random.randint(-2, 3, size=(n*m, target_rank)) / 2.0
            V = np.random.randint(-2, 3, size=(m*p, target_rank)) / 2.0
            W = np.random.randint(-2, 3, size=(p*n, target_rank)) / 2.0
            
            res = tensor_residual(U, V, W)
            if res < 1e-4:
                if verify_solution(U, V, W, 1e-6):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    return best
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                if verify_solution(U_r, V_r, W_r, 1e-6):
                    best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = target_rank
                    return best
    
    return best
# EVOLVE-BLOCK-END
