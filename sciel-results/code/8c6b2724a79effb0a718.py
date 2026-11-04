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
    
    def check_exact(U_r, V_r, W_r, tol=1e-6):
        """Check if the decomposition is exact."""
        recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        return np.max(np.abs(recon - T)) < tol
    
    def objective(params, rank):
        """Objective function: sum of squared errors."""
        U_r = params[:u_dim * rank].reshape(u_dim, rank)
        V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
        W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
        recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        diff = recon.flatten() - T_flat
        return np.dot(diff, diff)
    
    def try_rank_structured(rank, num_trials=40):
        """
        Try to find a rank-r decomposition using structured initialization.
        
        Key insight: For 2x3x3, we can group terms by which output row they
        contribute to. Each output row i_c uses 3 input columns j_a, so we
        can try to share computation across the two output rows.
        """
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            strategy = trial % 6
            
            if strategy == 0:
                # Structured: separate blocks for each output row
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                
                terms_per_row = rank // n
                for i_c in range(n):
                    for t in range(terms_per_row):
                        c = i_c * terms_per_row + t
                        if c >= rank:
                            break
                        # A factors: only row i_c of A contributes
                        for j_a in range(m):
                            U_init[i_c * m + j_a, c] = 1.0
                        # B factors: all of B
                        for j_a in range(m):
                            for k_b in range(p):
                                V_init[j_a * p + k_b, c] = 1.0 / m
                        # C factors: only row i_c of C
                        for k_b in range(p):
                            W_init[k_b * n + i_c, c] = 1.0
                
            elif strategy == 1:
                # Winograd-style: try to combine terms
                # For each j_a, we need to compute sum over k_b
                # Try to use fewer than m*p terms by exploiting linearity
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                
                # Assign terms cyclically
                np.random.seed(100 + trial * 7)
                for c in range(rank):
                    # Random pattern for U
                    for i_a in range(n):
                        for j_a in range(m):
                            U_init[i_a * m + j_a, c] = np.random.choice([-1, 0, 1])
                    # Random pattern for V  
                    for j_a in range(m):
                        for k_b in range(p):
                            V_init[j_a * p + k_b, c] = np.random.choice([-0.5, 0, 0.5])
                    # Random pattern for W
                    for k_b in range(p):
                        for i_c in range(n):
                            W_init[k_b * n + i_c, c] = np.random.choice([-1, 0, 1])
                
            elif strategy == 2:
                # Block diagonal structure: group by j_a
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                
                terms_per_j = max(1, rank // m)
                for j_a in range(m):
                    for t in range(terms_per_j):
                        c = j_a * terms_per_j + t
                        if c >= rank:
                            break
                        # U: both rows of A, column j_a
                        U_init[0 * m + j_a, c] = 1.0
                        U_init[1 * m + j_a, c] = 1.0
                        # V: column j_a of B, all rows
                        for k_b in range(p):
                            V_init[j_a * p + k_b, c] = 1.0
                        # W: all of C
                        for k_b in range(p):
                            for i_c in range(n):
                                W_init[k_b * n + i_c, c] = 1.0 / p
                
            elif strategy == 3:
                # Perturbed schoolbook: start from rank-18 and try to merge
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                
                # Initialize with first 'rank' terms of schoolbook
                for c in range(min(rank, 18)):
                    i_a = c // (m * p)
                    j_a = (c // p) % m
                    k_b = c % p
                    if i_a < n and j_a < m and k_b < p:
                        U_init[i_a * m + j_a, c] = 1.0
                        V_init[j_a * p + k_b, c] = 1.0
                        W_init[k_b * n + i_a, c] = 1.0
                
                # Add random perturbation to encourage merging
                np.random.seed(200 + trial * 11)
                U_init += np.random.randn(u_dim, rank) * 0.05
                V_init += np.random.randn(v_dim, rank) * 0.05
                W_init += np.random.randn(w_dim, rank) * 0.05
                
            elif strategy == 4:
                # Symmetric structure: try to make U and W related
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                
                np.random.seed(300 + trial * 13)
                for c in range(rank):
                    # Generate random base vectors
                    u_vec = np.random.randn(u_dim)
                    v_vec = np.random.randn(v_dim)
                    w_vec = np.random.randn(w_dim)
                    
                    # Scale to encourage sparsity
                    u_vec[np.abs(u_vec) < 0.5] = 0
                    v_vec[np.abs(v_vec) < 0.5] = 0
                    w_vec[np.abs(w_vec) < 0.5] = 0
                    
                    U_init[:, c] = u_vec
                    V_init[:, c] = v_vec
                    W_init[:, c] = w_vec
                    
            else:
                # Hybrid: mix of structured and random
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                
                # First half: structured by output row
                terms_per_row = rank // (2 * n)
                for i_c in range(n):
                    for t in range(terms_per_row):
                        c = i_c * terms_per_row + t
                        if c >= rank:
                            break
                        for j_a in range(m):
                            U_init[i_c * m + j_a, c] = 1.0
                        for j_a in range(m):
                            for k_b in range(p):
                                V_init[j_a * p + k_b, c] = 1.0 / m
                        for k_b in range(p):
                            W_init[k_b * n + i_c, c] = 1.0
                
                # Second half: random
                np.random.seed(400 + trial * 17)
                start_c = terms_per_row * n
                for c in range(start_c, rank):
                    U_init[:, c] = np.random.randn(u_dim) * 0.3
                    V_init[:, c] = np.random.randn(v_dim) * 0.3
                    W_init[:, c] = np.random.randn(w_dim) * 0.3
                    U_init[np.abs(U_init[:, c]) < 0.1, c] = 0
                    V_init[np.abs(V_init[:, c]) < 0.1, c] = 0
                    W_init[np.abs(W_init[:, c]) < 0.1, c] = 0
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Phase 1: Coarse optimization
            method_choice = trial % 4
            if method_choice == 0:
                method = 'L-BFGS-B'
                options = {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            elif method_choice == 1:
                method = 'Nelder-Mead'
                options = {'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            elif method_choice == 2:
                method = 'Powell'
                options = {'maxiter': 300, 'ftol': 1e-10, 'xtol': 1e-8}
            else:
                # Use COBYLA for unconstrained
                method = 'COBYLA'
                options = {'maxiter': 500, 'rhobeg': 1e-4}
            
            try:
                result = minimize(lambda x: objective(x, rank), params_init, method=method, options=options)
            except Exception:
                continue
            
            if result.fun < 1e-8:
                params = result.x
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                
                if check_exact(U_r, V_r, W_r, tol=1e-6):
                    # Phase 2: Polish with tighter tolerance
                    try:
                        result2 = minimize(lambda x: objective(x, rank), params, method='L-BFGS-B', 
                                         options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
                        if result2.fun < 1e-14:
                            params = result2.x
                            U_r = params[:u_dim * rank].reshape(u_dim, rank)
                            V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                            W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                    except Exception:
                        pass
                    
                    if check_exact(U_r, V_r, W_r, tol=1e-6):
                        # Try to round to half-integers
                        U_round = np.round(U_r * 2) / 2
                        V_round = np.round(V_r * 2) / 2
                        W_round = np.round(W_r * 2) / 2
                        if check_exact(U_round, V_round, W_round, tol=1e-6):
                            return U_round, V_round, W_round
                        return U_r, V_r, W_r
        
        return None
    
    # Try ranks from 12 up to 17 (schoolbook is 18)
    # Focus on the most likely candidates first
    candidate_ranks = [12, 13, 14, 15, 16, 17]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        result = try_rank_structured(rank, num_trials=35)
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
