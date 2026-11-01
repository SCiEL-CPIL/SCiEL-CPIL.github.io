from scipy.optimize import minimize
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
        
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school

    # Target rank to try: 10
    target_rank = 10
    
    # Time management
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit to be safe

    def get_tensor(U, V, W):
        """
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        Returns the full tensor T'_{a,b,c}
        """
        rank = U.shape[1]
        T_prime = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u_r = np.array(U[:, r])
            v_r = np.array(V[:, r])
            w_r = np.array(W[:, r])
            T_prime += np.einsum('i,j,k->ijk', u_r, v_r, w_r)
        return T_prime

    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = get_target_tensor()

    # Define objective function for optimization
    def objective(params):
        if time.time() > deadline:
            return 1e9
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
        W_flat = params[n*m*target_rank + m*p*target_rank:]
        
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        
        T_prime = get_tensor(U, V, W)
        diff = T_prime - T_target
        return np.sum(diff**2)

    # Try to find a rank 10 solution
    random.seed(42)
    found_valid_rank_10 = False
    
    # We will try a few random initializations
    for attempt in range(20):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize with small random values
        init_size = (n*m + m*p + p*n) * target_rank
        params0 = np.random.randn(init_size) * 0.5
        
        try:
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < 1e-6:
                U_flat = res.x[:n*m*target_rank]
                V_flat = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                W_flat = res.x[n*m*target_rank + m*p*target_rank:]
                
                U_num = U_flat.reshape(n*m, target_rank)
                V_num = V_flat.reshape(m*p, target_rank)
                W_num = W_flat.reshape(p*n, target_rank)
                
                # Round to nearest 0.5
                U_rnd = np.round(U_num * 2) / 2
                V_rnd = np.round(V_num * 2) / 2
                W_rnd = np.round(W_num * 2) / 2
                
                # Verify exactness
                T_rnd = get_tensor(U_rnd, V_rnd, W_rnd)
                if np.allclose(T_rnd, T_target, atol=1e-9):
                    # Convert to lists
                    best_U = U_rnd.tolist()
                    best_V = V_rnd.tolist()
                    best_W = W_rnd.tolist()
                    best_rank = target_rank
                    found_valid_rank_10 = True
                    break
        except Exception:
            continue

    return {"U": best_U, "V": best_V, "W": best_W}
