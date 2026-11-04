import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    trivial_rank = n * m * p  # 24
    
    # Trivial solution for 2x3x4
    U_trivial = np.zeros((nm, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((mp, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((pn, trivial_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
                
    best_rank = trivial_rank
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    
    def check_valid_232(U, V, W, rank, tol=1e-9):
        # n=2, m=3, p=2
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*3 + j, r] * V[j*2 + k, r] * W[k*2 + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    # Try to find a solution for 2x3x2 with rank 9, 10, 11
    found_small = False
    for target_rank_small in [9, 10, 11]:
        if time.time() > deadline - 30.0:
            break
            
        nm_s = 2 * 3 # 6
        mp_s = 3 * 2 # 6
        pn_s = 2 * 2 # 4
        
        np.random.seed(42)
        U_init = np.random.rand(nm_s, target_rank_small) * 0.5 - 0.25
        V_init = np.random.rand(mp_s, target_rank_small) * 0.5 - 0.25
        W_init = np.random.rand(pn_s, target_rank_small) * 0.5 - 0.25
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def error_func_small(x):
            U = x[:nm_s*target_rank_small].reshape(nm_s, target_rank_small)
            V = x[nm_s*target_rank_small:nm_s*target_rank_small+mp_s*target_rank_small].reshape(mp_s, target_rank_small)
            W = x[nm_s*target_rank_small+mp_s*target_rank_small:].reshape(pn_s, target_rank_small)
            
            err = 0.0
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        s = 0.0
                        for r in range(target_rank_small):
                            s += U[i*3 + j, r] * V[j*2 + k, r] * W[k*2 + i, r]
                        err += (s - 1.0)**2
            return err
            
        try:
            res = minimize(error_func_small, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 10000})
            if res.success and res.fun < 1e-9:
                U_sol = res.x[:nm_s*target_rank_small].reshape(nm_s, target_rank_small)
                V_sol = res.x[nm_s*target_rank_small:nm_s*target_rank_small+mp_s*target_rank_small].reshape(mp_s, target_rank_small)
                W_sol = res.x[nm_s*target_rank_small+mp_s*target_rank_small:].reshape(pn_s, target_rank_small)
                
                U_s, V_s, W_s = snap_to_half_integer(U_sol, V_sol, W_sol)
                if check_valid_232(U_s, V_s, W_s, target_rank_small):
                    # Construct 2x3x4 solution
                    rank_total = 2 * target_rank_small
                    U_total = np.zeros((nm, rank_total), dtype=np.float64)
                    V_total = np.zeros((mp, rank_total), dtype=np.float64)
                    W_total = np.zeros((pn, rank_total), dtype=np.float64)
                    
                    # First half: k=0,1
                    U_total[:, 0:target_rank_small] = U_s
                    V_total[:, 0:target_rank_small] = V_s
                    W_total[:, 0:target_rank_small] = W_s # W indices for k=0,1 are 0..3
                    
                    # Second half: k=2,3
                    # W indices for k=2,3 are 4..7
                    # We need to map W_s (which has indices 0..3 for k=0,1) to 4..7
                    W_second = np.zeros_like(W_s)
                    # W_s[k*2+i, r] corresponds to k=0,1
                    # We need W_total[k*2+i, r] for k=2,3
                    # So W_total[4+i, r] should be W_s[0+i, r]? No.
                    # W_s is for p=2. Indices 0,1,2,3 correspond to (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1).
                    # W_total is for p=4. Indices 0..7.
                    # For the second block, we are computing C[:, 2:4] = A @ B[:, 2:4].
                    # This is equivalent to 2x3x2 multiplication.
                    # The W matrix for this block will have indices corresponding to (k=2,i=0), (k=2,i=1), (k=3,i=0), (k=3,i=1).
                    # These are indices 4,5,6,7 in W_total.
                    # The structure of the decomposition is the same, so we can reuse W_s values.
                    W_total[4:8, target_rank_small:2*target_rank_small] = W_s
                    V_total[:, target_rank_small:2*target_rank_small] = V_s
                    U_total[:, target_rank_small:2*target_rank_small] = U_s
                    
                    if check_valid(U_total, V_total, W_total, rank_total, tol=1e-9):
                         best_rank = rank_total
                         best_U = U_total
                         best_V = V_total
                         best_W = W_total
                         found_small = True
                         break
        except Exception:
            pass
             
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
