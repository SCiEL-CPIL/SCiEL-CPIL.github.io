import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1

    def check_solution(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        if np.linalg.norm(T_calc - T_target) < 1e-6:
            # Try to round
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.linalg.norm(T_check - T_target) < 1e-6:
                return U_r, V_r, W_r
        return None

    start_time = time.time()
    deadline = start_time + 110

    # Try to find rank 74
    # Strategy: Remove one column from schoolbook and optimize
    # We will try a few random columns to remove
    
    import random
    cols_to_try = random.sample(range(rank_sb), 10)
    
    for c_remove in cols_to_try:
        if time.time() > deadline:
            break
            
        # Create initial guess
        U_init = np.delete(U_sb, c_remove, axis=1)
        V_init = np.delete(V_sb, c_remove, axis=1)
        W_init = np.delete(W_sb, c_remove, axis=1)
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Optimize
        for step in range(200):
            if time.time() > deadline:
                break
                
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_calc - T_target
            err = np.sum(diff**2)
            
            if err < 1e-8:
                sol = check_solution(U, V, W)
                if sol:
                    U_sol, V_sol, W_sol = sol
                    return {
                        "U": U_sol.tolist(),
                        "V": V_sol.tolist(),
                        "W": W_sol.tolist()
                    }
                
            gU = np.einsum('abc,br,cr->ar', diff, V, W)
            gV = np.einsum('abc,ar,cr->br', diff, U, W)
            gW = np.einsum('abc,ar,br->cr', diff, U, V)
            
            lr = 0.05
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
