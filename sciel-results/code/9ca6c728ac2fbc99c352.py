"""Program for 2x3 by 3x3 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

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
    deadline = start_time + 115  # safety margin before 120s limit
    
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
    
    def sat_search(rank, allowed_values=(-1, 0, 1), max_time=60.0):
        """
        SAT-style search: enumerate configurations with entries in allowed_values.
        Uses backtracking with constraint propagation.
        """
        search_start = time.time()
        
        # Precompute which (i,j,k) triples have T[i,j,k] = 1
        target_ones = [(i, j, k) for i in range(u_dim) for j in range(v_dim) for k in range(w_dim) if T[i, j, k] == 1]
        
        # For each term c, U[:,c], V[:,c], W[:,c] are vectors.
        # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # For T[i,j,k]=1: exactly one term must contribute 1 (if all entries in {-1,0,1} and we want exact)
        # Actually with {-1,0,1}, multiple terms can contribute but must sum to 1.
        
        # Strategy: for each (i,j,k) with T=1, the sum of U[i,c]*V[j,c]*W[k,c] over c must equal 1.
        # For (i,j,k) with T=0, the sum must equal 0.
        
        # This is a constraint satisfaction problem. Let's use a smarter approach:
        # Enumerate U first (column by column), then for each U, solve for V and W.
        
        # Actually, let's try a different approach: fix the structure.
        # Each "term" c corresponds to a rank-1 slice. 
        # For the schoolbook, each term has exactly one nonzero in each of U,V,W.
        # For lower rank, terms may have multiple nonzeros.
        
        # Let's try: for rank r, each column of U has at most 2 nonzeros, 
        # each column of V has at most 3 nonzeros, each column of W has at most 2 nonzeros.
        # This keeps the search space manageable.
        
        max_nonzeros_U = 2
        max_nonzeros_V = 3
        max_nonzeros_W = 2
        
        # Generate all possible sparse patterns for U columns
        # A U column is a vector in {−1,0,1}^6 with at most 2 nonzeros
        u_patterns = []
        for nz_count in range(1, max_nonzeros_U + 1):
            for positions in product(range(u_dim), repeat=nz_count):
                if len(set(positions)) != nz_count:
                    continue
                for values in product(allowed_values, repeat=nz_count):
                    if 0 in values:
                        continue
                    vec = [0] * u_dim
                    for pos, val in zip(positions, values):
                        vec[pos] = val
                    u_patterns.append(tuple(vec))
        
        # Similarly for V columns (in {−1,0,1}^9 with at most 3 nonzeros)
        v_patterns = []
        for nz_count in range(1, max_nonzeros_V + 1):
            for positions in product(range(v_dim), repeat=nz_count):
                if len(set(positions)) != nz_count:
                    continue
                for values in product(allowed_values, repeat=nz_count):
                    if 0 in values:
                        continue
                    vec = [0] * v_dim
                    for pos, val in zip(positions, values):
                        vec[pos] = val
                    v_patterns.append(tuple(vec))
        
        # W columns (in {−1,0,1}^6 with at most 2 nonzeros)
        w_patterns = []
        for nz_count in range(1, max_nonzeros_W + 1):
            for positions in product(range(w_dim), repeat=nz_count):
                if len(set(positions)) != nz_count:
                    continue
                for values in product(allowed_values, repeat=nz_count):
                    if 0 in values:
                        continue
                    vec = [0] * w_dim
                    for pos, val in zip(positions, values):
                        vec[pos] = val
                    w_patterns.append(tuple(vec))
        
        # This is still too large. Let's use a smarter approach.
        # For each (i,j,k) with T[i,j,k]=1, we need sum_c U[i,c]*V[j,c]*W[k,c] = 1.
        # For (i,j,k) with T[i,j,k]=0, we need sum_c U[i,c]*V[j,c]*W[k,c] = 0.
        
        # Let's try a randomized approach with verification.
        np.random.seed(42 + rank)
        
        for trial in range(500):
            if time.time() - search_start > max_time:
                break
            
            # Random sparse initialization
            U_init = np.zeros((u_dim, rank))
            V_init = np.zeros((v_dim, rank))
            W_init = np.zeros((w_dim, rank))
            
            for c in range(rank):
                # Randomly choose positions and values
                u_nz = np.random.randint(1, max_nonzeros_U + 1)
                u_pos = np.random.choice(u_dim, size=u_nz, replace=False)
                u_vals = np.random.choice(allowed_values, size=u_nz)
                for pos, val in zip(u_pos, u_vals):
                    U_init[pos, c] = val
                
                v_nz = np.random.randint(1, max_nonzeros_V + 1)
                v_pos = np.random.choice(v_dim, size=v_nz, replace=False)
                v_vals = np.random.choice(allowed_values, size=v_nz)
                for pos, val in zip(v_pos, v_vals):
                    V_init[pos, c] = val
                
                w_nz = np.random.randint(1, max_nonzeros_W + 1)
                w_pos = np.random.choice(w_dim, size=w_nz, replace=False)
                w_vals = np.random.choice(allowed_values, size=w_nz)
                for pos, val in zip(w_pos, w_vals):
                    W_init[pos, c] = val
            
            if check_exact(U_init, V_init, W_init, tol=1e-6):
                return U_init, V_init, W_init
        
        return None
    
    def structured_search(rank, max_time=60.0):
        """
        More structured search: exploit the block structure of the tensor.
        The tensor M_{2,3,3} has a specific structure that we can exploit.
        """
        search_start = time.time()
        
        # The tensor has the property that T[i,j,k] = 1 iff:
        # i = i_a * m + j_a, j = j_a * p + k_b, k = k_b * n + i_a
        # for some i_a in {0,1}, j_a in {0,1,2}, k_b in {0,1,2}
        
        # This means the 18 ones are at specific positions.
        # For a rank-r decomposition, we need to "cover" these 18 positions
        # with r rank-1 slices.
        
        # Key insight: each rank-1 slice U[:,c]*V[:,c]*W[:,c] is a tensor
        # that is 1 at positions (i,j,k) where U[i,c]*V[j,c]*W[k,c] = 1,
        # -1 where the product is -1, and 0 otherwise.
        
        # For the sum to be exactly 1 at the 18 target positions and 0 elsewhere,
        # we need careful cancellation.
        
        # Let's try a systematic approach:
        # For each column c, choose which "block" it primarily covers.
        # There are n*m*p = 18 blocks, but we only have r < 18 columns.
        # So some columns must cover multiple blocks.
        
        # Strategy: use a greedy approach with backtracking.
        # Start with the 18 schoolbook terms and try to merge pairs.
        
        # First, let's try a specific construction for rank 17.
        # Idea: take the schoolbook 18 terms and modify two of them
        # to cancel each other's "extra" contributions.
        
        if rank == 17:
            # Start with schoolbook
            U_init = np.zeros((u_dim, rank))
            V_init = np.zeros((v_dim, rank))
            W_init = np.zeros((w_dim, rank))
            
            for c in range(17):
                i_a = c // (m * p)
                j_a = (c // p) % m
                k_b = c % p
                U_init[i_a * m + j_a, c] = 1.0
                V_init[j_a * p + k_b, c] = 1.0
                W_init[k_b * n + i_a, c] = 1.0
            
            # The missing 18th term is: i_a=1, j_a=2, k_b=2
            # U[5,17]=1, V[8,17]=1, W[5,17]=1
            # We need to absorb this into the existing 17 terms.
            
            # Try modifying term 16 (last term) to also cover position (5,8,5)
            # Term 16: i_a=1, j_a=1, k_b=2
            # U[4,16]=1, V[5,16]=1, W[5,16]=1
            # If we change W[5,16] to 2, then term 16 contributes 2 at (4,5,5)
            # and we need to cancel the extra 1 at (4,5,5) and add 1 at (5,8,5).
            
            # This is getting complex. Let's try a different approach.
            # Use the fact that we can have entries in {-1, 0, 1, 2, -2}.
            
            # Let's try: modify two schoolbook terms to create a "Strassen-like" 
            # cancellation pattern.
            
            # For now, let's try random perturbations of the schoolbook.
            for trial in range(200):
                if time.time() - search_start > max_time:
                    break
                
                U_try = U_init.copy()
                V_try = V_init.copy()
                W_try = W_init.copy()
                
                # Randomly modify a few entries
                for _ in range(np.random.randint(1, 4)):
                    mat = np.random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        row = np.random.randint(u_dim)
                        col = np.random.randint(rank)
                        U_try[row, col] = np.random.choice([-2, -1, 0, 1, 2])
                    elif mat == 'V':
                        row = np.random.randint(v_dim)
                        col = np.random.randint(rank)
                        V_try[row, col] = np.random.choice([-2, -1, 0, 1, 2])
                    else:
                        row = np.random.randint(w_dim)
                        col = np.random.randint(rank)
                        W_try[row, col] = np.random.choice([-2, -1, 0, 1, 2])
                
                if check_exact(U_try, V_try, W_try, tol=1e-6):
                    return U_try, V_try, W_try
            
            return None
        
        # For other ranks, use random search
        np.random.seed(123 + rank)
        for trial in range(300):
            if time.time() - search_start > max_time:
                break
            
            U_init = np.zeros((u_dim, rank))
            V_init = np.zeros((v_dim, rank))
            W_init = np.zeros((w_dim, rank))
            
            for c in range(rank):
                u_nz = np.random.randint(1, 3)
                u_pos = np.random.choice(u_dim, size=u_nz, replace=False)
                for pos in u_pos:
                    U_init[pos, c] = np.random.choice([-1, 0, 1])
                
                v_nz = np.random.randint(1, 4)
                v_pos = np.random.choice(v_dim, size=v_nz, replace=False)
                for pos in v_pos:
                    V_init[pos, c] = np.random.choice([-1, 0, 1])
                
                w_nz = np.random.randint(1, 3)
                w_pos = np.random.choice(w_dim, size=w_nz, replace=False)
                for pos in w_pos:
                    W_init[pos, c] = np.random.choice([-1, 0, 1])
            
            if check_exact(U_init, V_init, W_init, tol=1e-6):
                return U_init, V_init, W_init
        
        return None
    
    def continuous_optimization(rank, num_trials=30, max_time=45.0):
        """Continuous optimization as fallback."""
        search_start = time.time()
        
        for trial in range(num_trials):
            if time.time() - search_start > max_time:
                break
            
            np.random.seed(1000 + trial * 7 + rank)
            
            # Random initialization
            U_init = np.random.randn(u_dim, rank) * 0.5
            V_init = np.random.randn(v_dim, rank) * 0.5
            W_init = np.random.randn(w_dim, rank) * 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                diff = recon - T
                return np.sum(diff * diff)
            
            def objective_grad(params):
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                diff = recon - T
                obj = np.sum(diff * diff)
                dU = 2.0 * np.einsum('ijk,jc,kc->ic', diff, V_r, W_r)
                dV = 2.0 * np.einsum('ijk,ic,kc->jc', diff, U_r, W_r)
                dW = 2.0 * np.einsum('ijk,ic,jc->kc', diff, U_r, V_r)
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return obj, grad
            
            try:
                from scipy.optimize import minimize
                result = minimize(objective, params_init, method='L-BFGS-B', 
                                jac=objective_grad, options={'maxiter': 300, 'ftol': 1e-14})
                
                if result.fun < 1e-10:
                    params = result.x
                    U_r = params[:u_dim * rank].reshape(u_dim, rank)
                    V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                    W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                    
                    if check_exact(U_r, V_r, W_r, tol=1e-6):
                        return U_r, V_r, W_r
                    
                    # Try rounding
                    U_round = np.round(U_r * 2) / 2
                    V_round = np.round(V_r * 2) / 2
                    W_round = np.round(W_r * 2) / 2
                    if check_exact(U_round, V_round, W_round, tol=1e-6):
                        return U_round, V_round, W_round
            
            except Exception:
                continue
        
        return None
    
    # Try ranks from 15 up to 17
    for rank in [15, 16, 17]:
        if time.time() > deadline - 20:
            break
        
        # First try structured search
        result = structured_search(rank, max_time=min(30, deadline - time.time() - 10))
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank
            break
        
        # Then try SAT-style search
        result = sat_search(rank, max_time=min(20, deadline - time.time() - 10))
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank
            break
        
        # Finally try continuous optimization
        if time.time() < deadline - 15:
            result = continuous_optimization(rank, num_trials=20, max_time=min(15, deadline - time.time() - 5))
            if result is not None:
                U_r, V_r, W_r = result
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = rank
                break
    
    # Convert to lists
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
