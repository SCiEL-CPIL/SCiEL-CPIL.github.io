import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = N, M, P
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6
    
    # Build the full tensor T[i,j,k]
    T = np.zeros((u_dim, v_dim, w_dim))
    for i in range(u_dim):
        i_a = i // m
        j_a = i % m
        for j in range(v_dim):
            j_a2 = j // p
            k_b = j % p
            if j_a != j_a2:
                continue
            for k in range(w_dim):
                k_b2 = k // n
                i_c = k % n
                if i_a != i_c or k_b != k_b2:
                    continue
                T[i, j, k] = 1.0

    # Schoolbook decomposition as baseline (Rank 18)
    rank_school = n * m * p
    U_s = np.zeros((u_dim, rank_school))
    V_s = np.zeros((v_dim, rank_school))
    W_s = np.zeros((w_dim, rank_school))
    col = 0
    for i_a in range(n):
        for j_a in range(m):
            for k_b in range(p):
            # Note: The index mapping in the previous code was:
            # i_idx = i_a * m + j_a
            # j_idx = j_a * p + k_b
            # k_idx = k_b * n + i_a
            # This matches the task description:
            # U[i*n+j, c], V[j*p+k, c], W[k*n+i, c]
                i_idx = i_a * m + j_a
                j_idx = j_a * p + k_b
                k_idx = k_b * n + i_a
                U_s[i_idx, col] = 1.0
                V_s[j_idx, col] = 1.0
                W_s[k_idx, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = rank_school
    
    # Try to find rank 17 decomposition using integer search
    target_rank = 17
    
    def try_rank_integer_search(rank):
        if time.time() > deadline:
            return None
            
        # We search for integer matrices U', V', W' such that 
        # sum_c U'[i,c] V'[j,c] W'[k,c] = 8 * T[i,j,k]
        # This allows U, V, W to be half-integers (divide by 2).
        # Actually, if we find integer U', V', W', then U = U'/2 etc.
        # The product is (U'/2)(V'/2)(W'/2) = U'V'W'/8.
        # So we need sum U'V'W' = 8T.
        
        T_target = 8.0 * T
        T_flat = T_target.flatten()
        
        def error_func(params):
            U_r = params[:u_dim * rank].reshape(u_dim, rank)
            V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
            W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
            recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
            diff = recon.flatten() - T_flat
            return np.dot(diff, diff)
        
        # Use random integer initialization in range [-3, 3]
        for trial in range(200):
            if time.time() > deadline:
                return None
                
            np.random.seed(1000 + trial)
            # Initialize with small integers
            U_init = np.random.randint(-3, 4, (u_dim, rank))
            V_init = np.random.randint(-3, 4, (v_dim, rank))
            W_init = np.random.randint(-3, 4, (w_dim, rank))
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use Nelder-Mead for derivative-free optimization on the integer grid?
            # No, Nelder-Mead works on continuous space.
            # We can use it to find a continuous solution, then round?
            # Or use it to minimize the error, and check if the solution is close to integers.
            
            try:
                result = minimize(error_func, params_init, method='Nelder-Mead', 
                                  options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            except Exception:
                continue
                
            if result.fun < 1e-6:
                params = result.x
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                
                # Check if the solution is close to half-integers
                # Multiply by 2 to check if integers
                U_check = 2.0 * U_r
                V_check = 2.0 * V_r
                W_check = 2.0 * W_r
                
                # Round to nearest integer
                U_rounded = np.round(U_check)
                V_rounded = np.round(V_check)
                W_rounded = np.round(W_check)
                
                # Verify the rounded solution
                recon_rounded = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(recon_rounded - T_target)) < 1e-5:
                    # Valid half-integer solution found
                    return U_rounded / 2.0, V_rounded / 2.0, W_rounded / 2.0
        return None

    result = try_rank_integer_search(target_rank)
    if result:
        best_U, best_V, best_W = result
        best_rank = target_rank
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
