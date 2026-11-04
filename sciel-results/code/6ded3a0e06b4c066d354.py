"""Program for 2x3 by 3x3 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The tensor T is of size (n*m) x (m*p) x (p*n) = 6 x 9 x 6.
    We seek rank-r decomposition: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    Returns dict with U (6 x r), V (9 x r), W (6 x r) where r is the rank.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = N, M, P
    # Tensor dimensions: U has n*m rows, V has m*p rows, W has p*n rows
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6
    
    # Build the full tensor T[i,j,k] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
    # T[i,j,k] = 1 if the corresponding matrix multiplication term is present
    # i = i_a * m + j_a where i_a is row of A, j_a is col of A
    # j = j_a * p + k_b where j_a is col of A (same as j_a above), k_b is col of B
    # k = k_b * n + i_c where k_b is row of B, i_c is row of C
    # The product C[i_c, k_b] = sum_{j_a} A[i_a=i_c, j_a] * B[j_a, k_b]
    # So T[i,j,k] = 1 iff i_a == i_c AND j_a == j_a (always) AND j_a (from V) == k_b's row... 
    # Let me be more careful.
    # i indexes (i_a, j_a) for A: i = i_a * m + j_a
    # j indexes (j_a, k_b) for B: j = j_a * p + k_b  
    # k indexes (k_b, i_c) for C: k = k_b * n + i_c
    # The tensor element T[i, j, k] = 1 iff j_a from i equals j_a from j, and i_a from i equals i_c from k
    # i.e., T[i,j,k] = 1 iff (i // m) == (k % n) and (i % m) == (j // p)
    
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
    
    # Try ranks from lower bound up
    # Lower bound: max(u_dim, v_dim, w_dim) = 9? Actually the tensor rank is at least the max of any matricization rank.
    # Let's try from rank 12 up (schoolbook is 18)
    
    best_rank = n * m * p  # fallback: schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_s = np.zeros((u_dim, rank_school))
    V_s = np.zeros((v_dim, rank_school))
    W_s = np.zeros((w_dim, rank_school))
    col = 0
    for i_a in range(n):
        for j_a in range(m):
            for k_b in range(p):
                i_idx = i_a * m + j_a
                j_idx = j_a * p + k_b
                k_idx = k_b * n + i_a
                U_s[i_idx, col] = 1.0
                V_s[j_idx, col] = 1.0
                W_s[k_idx, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # We'll use a randomized approach: try various ranks and use optimization
    
    for rank in range(12, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(42 + trial * 1000 + rank)
            U_init = np.random.randn(u_dim, rank)
            V_init = np.random.randn(v_dim, rank)
            W_init = np.random.randn(w_dim, rank)
            
            # Flatten the tensor
            T_flat = T.flatten()
            
            # Define the objective: minimize ||T - U*V*W||^2
            def objective(params):
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                # Compute the tensor product
                # T_recon[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
                # This is a 3-way tensor product
                # Efficient computation: for each c, add U[:,c] outer V[:,c] outer W[:,c]
                recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                diff = recon.flatten() - T_flat
                return np.dot(diff, diff)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, params_init, method='L-BFGS-B', 
                                  options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            except Exception:
                continue
            
            if result.fun < 1e-8:
                # Check if it's truly exact
                params = result.x
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                if np.max(np.abs(recon - T)) < 1e-6:
                    # Found a valid decomposition
                    # Round to nearest half-integer if possible
                    best_rank = rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break
        
        if best_rank == rank:
            break
    
    # Convert to lists of lists with integer/half-integer entries if possible
    def to_half_integer(mat):
        """Round values to nearest half-integer."""
        rounded = np.round(mat * 2) / 2
        # Verify rounding didn't break the decomposition
        return rounded
    
    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)
    
    # Verify the rounded version still works
    recon = np.einsum('ic,jc,kc->ijk', U_out, V_out, W_out)
    if np.max(np.abs(recon - T)) > 1e-4:
        # Rounding broke it, use original
        U_out = best_U
        V_out = best_V
        W_out = best_W
    
    # Convert to Python lists
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
