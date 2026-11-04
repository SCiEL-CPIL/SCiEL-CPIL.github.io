import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank
    # Target tensor T
    # T[alpha, beta, gamma] = 1 if alpha=(i,j), beta=(j,k), gamma=(k,i)
    # Indices:
    # alpha in [0, n*m)
    # beta in [0, m*p)
    # gamma in [0, p*n)
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    def check_solution(U, V, W):
        # Check if U, V, W are valid half-integers and decompose T
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Check exactness
        # Compute tensor
        # T_comp[alpha, beta, gamma] = sum_c U[alpha, c] V[beta, c] W[gamma, c]
        # This is a batched matrix multiplication or einsum
        # T_comp = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
        # But dimensions are large? 9x12x12 is small.
        T_comp = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
        
        if not np.allclose(T_comp, T, atol=1e-9):
            return False
        return True

    # Try ranks 33, 34, 35
    # Time limit 150s.
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    for r in [33, 34, 35]:
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale matters
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Flatten
        x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def fun(x):
            if time.time() > deadline:
                return np.inf
            U_f = x[:n*m*r].reshape(n*m, r)
            V_f = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_f = x[n*m*r + m*p*r :].reshape(p*n, r)
            T_comp = np.einsum('ac,bc,dc->abd', U_f, V_f, W_f)
            return (T_comp - T).flatten()
        
        # Use scipy.optimize.least_squares
        try:
            import scipy.optimize as opt
            res = opt.least_squares(fun, x, method='lm', max_nfev=1000)
            if time.time() > deadline:
                break
                
            x_sol = res.x
            U_sol = x_sol[:n*m*r].reshape(n*m, r)
            V_sol = x_sol[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_sol = x_sol[n*m*r + m*p*r :].reshape(p*n, r)
            
            if check_solution(U_sol, V_sol, W_sol):
                # Convert to lists
                U_list = U_sol.tolist()
                V_list = V_sol.tolist()
                W_list = W_sol.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
                
        except Exception as e:
            pass

    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
