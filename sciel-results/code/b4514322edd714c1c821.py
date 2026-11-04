"""Program for 2x3 by 3x3 matrix multiplication tensor decomposition."""
import time
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
    # i = i_a * m + j_a where i_a is row of A, j_a is col of A
    # j = j_a * p + k_b where j_a is col of A (same as j_a above), k_b is col of B
    # k = k_b * n + i_c where k_b is row of B, i_c is row of C
    # The product C[i_c, k_b] = sum_{j_a} A[i_a=i_c, j_a] * B[j_a, k_b]
    # So T[i,j,k] = 1 iff i_a == i_c AND j_a (from i) == j_a (from j) AND k_b (from j) == k_b (from k)
    # i.e., T[i,j,k] = 1 iff (i // m) == (k % n) and (i % m) == (j // p) and (j % p) == (k // n)
    
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
    
    T_flat = T.flatten()
    
    # Schoolbook decomposition as baseline (rank 18)
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
    
    # Try to find lower rank decompositions using structured search
    # We'll use a Winograd-inspired approach: try to find decompositions
    # where the factors have sparse/structured entries
    
    def check_exact(U_r, V_r, W_r, tol=1e-6):
        """Check if the decomposition is exact."""
        recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        return np.max(np.abs(recon - T)) < tol
    
    def try_rank(rank, num_trials=20):
        """Try to find a rank-r decomposition using multiple optimization strategies."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Different initialization strategies
            strategy = trial % 5
            
            if strategy == 0:
                # Pure random
                np.random.seed(1000 + trial * 7 + rank)
                U_init = np.random.randn(u_dim, rank)
                V_init = np.random.randn(v_dim, rank)
                W_init = np.random.randn(w_dim, rank)
            elif strategy == 1:
                # Sparse random (many zeros)
                np.random.seed(2000 + trial * 11 + rank)
                U_init = np.random.randn(u_dim, rank) * (np.random.rand(u_dim, rank) < 0.3)
                V_init = np.random.randn(v_dim, rank) * (np.random.rand(v_dim, rank) < 0.3)
                W_init = np.random.randn(w_dim, rank) * (np.random.rand(w_dim, rank) < 0.3)
            elif strategy == 2:
                # Based on partial schoolbook structure
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                for c in range(min(rank, 18)):
                    i_a = c // (m * p)
                    j_a = (c // p) % m
                    k_b = c % p
                    if i_a < n and j_a < m and k_b < p:
                        U_init[i_a * m + j_a, c] = 1.0
                        V_init[j_a * p + k_b, c] = 1.0
                        W_init[k_b * n + i_a, c] = 1.0
                # Add random perturbation
                np.random.seed(3000 + trial * 13 + rank)
                U_init += np.random.randn(u_dim, rank) * 0.1
                V_init += np.random.randn(v_dim, rank) * 0.1
                W_init += np.random.randn(w_dim, rank) * 0.1
            elif strategy == 3:
                # Block-structured: group terms by j_a
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                # Assign rank terms to different j_a groups
                terms_per_j = max(1, rank // m)
                for j_a in range(m):
                    for t in range(terms_per_j):
                        c = j_a * terms_per_j + t
                        if c >= rank:
                            break
                        # Random within the block
                        np.random.seed(4000 + j_a * 17 + t * 23 + rank)
                        rows_U = np.random.choice(u_dim, size=min(3, u_dim), replace=False)
                        rows_V = np.random.choice(v_dim, size=min(3, v_dim), replace=False)
                        rows_W = np.random.choice(w_dim, size=min(3, w_dim), replace=False)
                        for r in rows_U:
                            U_init[r, c] = np.random.choice([-1, 0, 1])
                        for r in rows_V:
                            V_init[r, c] = np.random.choice([-1, 0, 1])
                        for r in rows_W:
                            W_init[r, c] = np.random.choice([-1, 0, 1])
            else:
                # Winograd-like: try to exploit the structure
                # For 2x3x3, think of it as two 1x3x3 multiplications
                # Each 1x3x3 can potentially use fewer than 3 multiplications
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                
                # Try a structured approach where we combine terms
                # Group by output row i_c
                for i_c in range(n):
                    for t in range(rank // n):
                        c = i_c * (rank // n) + t
                        if c >= rank:
                            break
                        np.random.seed(5000 + i_c * 31 + t * 41 + rank)
                        # A factors: rows i_c*m + j_a for j_a in [0,m)
                        for j_a in range(m):
                            U_init[i_c * m + j_a, c] = np.random.choice([-1, 0, 1])
                        # B factors: cols j_a*p + k_b
                        for j_a in range(m):
                            for k_b in range(p):
                                V_init[j_a * p + k_b, c] = np.random.choice([-1, 0, 1]) * 0.5
                        # C factors: rows k_b*n + i_c
                        for k_b in range(p):
                            W_init[k_b * n + i_c, c] = np.random.choice([-1, 0, 1])
            
            # Flatten parameters
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                diff = recon.flatten() - T_flat
                return np.dot(diff, diff)
            
            # Use multiple optimization methods
            if trial % 3 == 0:
                method = 'L-BFGS-B'
                options = {'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            elif trial % 3 == 1:
                method = 'Nelder-Mead'
                options = {'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            else:
                method = 'Powell'
                options = {'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-10}
            
            try:
                result = minimize(objective, params_init, method=method, options=options)
            except Exception:
                continue
            
            if result.fun < 1e-10:
                params = result.x
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                
                if check_exact(U_r, V_r, W_r, tol=1e-6):
                    return U_r, V_r, W_r
                
                # Try to polish and round
                if result.fun < 1e-6:
                    # Attempt rounding to half-integers
                    U_round = np.round(U_r * 2) / 2
                    V_round = np.round(V_r * 2) / 2
                    W_round = np.round(W_r * 2) / 2
                    if check_exact(U_round, V_round, W_round, tol=1e-6):
                        return U_round, V_round, W_round
        
        return None
    
    # Try ranks from 12 up to 17 (schoolbook is 18)
    # Focus on the most likely candidates first
    candidate_ranks = [12, 13, 14, 15, 16, 17]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        result = try_rank(rank, num_trials=20)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank
            break  # Found the lowest rank, stop
    
    # If we found a lower rank, great. Otherwise use schoolbook.
    # Convert to lists with integer/half-integer entries if possible
    def to_half_integer(mat):
        """Round values to nearest half-integer and verify."""
        rounded = np.round(mat * 2) / 2
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
