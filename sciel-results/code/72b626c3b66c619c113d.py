import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Schoolbook decomposition (rank 12)
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school
    
    # Known rank-11 solution (Strassen + schoolbook for col 2)
    rank_known = 11
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    U_known[0][0] = 1; U_known[3][0] = 1
    V_known[0][0] = 1; V_known[4][0] = 1
    U_known[2][1] = 1; U_known[3][1] = 1
    V_known[0][1] = 1
    U_known[0][2] = 1
    V_known[1][2] = 1; V_known[4][2] = -1
    U_known[3][3] = 1
    V_known[3][3] = 1; V_known[0][3] = -1
    U_known[0][4] = 1; U_known[1][4] = 1
    V_known[4][4] = 1
    U_known[2][5] = 1; U_known[0][5] = -1
    V_known[0][5] = 1; V_known[1][5] = 1
    U_known[1][6] = 1; U_known[3][6] = -1
    V_known[3][6] = 1; V_known[4][6] = 1
    W_known[0][0] = 1; W_known[0][3] = 1; W_known[0][4] = -1; W_known[0][6] = 1
    W_known[2][2] = 1; W_known[2][4] = 1
    W_known[1][1] = 1; W_known[1][3] = 1
    W_known[3][0] = 1; W_known[3][1] = -1; W_known[3][2] = 1; W_known[3][5] = 1
    U_known[0][7] = 1
    V_known[2][7] = 1
    U_known[1][8] = 1
    V_known[5][8] = 1
    U_known[2][9] = 1
    V_known[2][9] = 1
    U_known[3][10] = 1
    V_known[5][10] = 1
    W_known[4][7] = 1; W_known[4][8] = 1
    W_known[5][9] = 1; W_known[5][10] = 1
    
    # Update best with rank-11 if valid
    U11_arr = np.array(U_known, dtype=float)
    V11_arr = np.array(V_known, dtype=float)
    W11_arr = np.array(W_known, dtype=float)
    
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
    T_target_flat = T_target.flatten()
    
    def compute_tensor_flat(U, V, W):
        """Compute flattened tensor from factor matrices"""
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return T.flatten()
    
    def verify(U, V, W):
        """Check if decomposition is exact"""
        T = compute_tensor_flat(U, V, W)
        return np.allclose(T, T_target_flat, atol=1e-9)
    
    if verify(U11_arr, V11_arr, W11_arr):
        best_U, best_V, best_W = U11_arr.tolist(), V11_arr.tolist(), W11_arr.tolist()
        best_rank = rank_known
    
    # Try to find rank-10 decomposition using numerical optimization
    target_rank = 10
    
    def objective(x):
        """Objective function for scipy.optimize.minimize"""
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U + size_V]
        W_flat = x[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        
        T_flat = compute_tensor_flat(U, V, W)
        diff = T_flat - T_target_flat
        return np.dot(diff, diff)
    
    found_rank10 = False
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with random small integers or random floats
        # Random integers in {-1, 0, 1} often work well for tensor ranks
        init_U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
        init_V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
        init_W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
        
        x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
            
            if res.fun < 1e-10:
                # Verify the solution
                x_opt = res.x
                size_U = n * m * target_rank
                size_V = m * p * target_rank
                size_W = p * n * target_rank
                
                U_flat = x_opt[:size_U]
                V_flat = x_opt[size_U:size_U + size_V]
                W_flat = x_opt[size_U + size_V:size_U + size_V + size_W]
                
                U_opt = U_flat.reshape(n * m, target_rank)
                V_opt = V_flat.reshape(m * p, target_rank)
                W_opt = W_flat.reshape(p * n, target_rank)
                
                if verify(U_opt, V_opt, W_opt):
                    # Round to nearest integer if close
                    U_opt = np.round(U_opt)
                    V_opt = np.round(V_opt)
                    W_opt = np.round(W_opt)
                    
                    if verify(U_opt, V_opt, W_opt):
                        best_U, best_V, best_W = U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
                        best_rank = target_rank
                        found_rank10 = True
                        break
        except Exception:
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
