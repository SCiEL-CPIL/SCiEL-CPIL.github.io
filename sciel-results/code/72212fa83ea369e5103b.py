import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n, m, p = 2, 4, 5
    
    # The tensor T has shape (n*m, m*p, p*n) = (8, 20, 10)
    # Indices:
    # a = i*m + j  (i in 0..n-1, j in 0..m-1)
    # b = j'*p + k (j' in 0..m-1, k in 0..p-1)
    # c = k'*n + i'(k' in 0..p-1, i' in 0..n-1)
    # T[a, b, c] = 1 if i==i', j==j', k==k' else 0
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Initialize the full schoolbook tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # We will try to find a decomposition of rank R.
    # Start with a reasonable upper bound, e.g., 30 (lower than 40).
    # We'll search for the minimal rank by trying to decompose into R rank-1 terms.
    
    # Strategy: 
    # 1. Initialize U, V, W with random vectors or specific structures.
    # 2. Use a greedy algorithm to pick the best rank-1 term to subtract.
    # 3. Or, use a more sophisticated approach: 
    #    Represent the tensor as a matrix unfolding.
    #    A decomposition of rank R is equivalent to finding matrices U (dim_u x R), V (dim_v x R), W (dim_w x R)
    #    such that T = sum_r U[:, r] V[:, r]^T W[:, r]^T (tensor outer product sum).
    
    # Let's try a "Greedy Rank-1 Subtraction" approach with randomization/restarts.
    # This is a heuristic search.
    
    best_rank = 40
    best_decomp = None
    
    # Initialize the known schoolbook decomposition as the starting "best"
    U_best = np.zeros((dim_u, 40))
    V_best = np.zeros((dim_v, 40))
    W_best = np.zeros((dim_w, 40))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_best[a, idx] = 1
                V_best[b, idx] = 1
                W_best[c, idx] = 1
                idx += 1
    best_decomp = (U_best, V_best, W_best)
    
    # Search for lower ranks
    # We'll attempt ranks from 30 down to 20.
    # For each rank R, we try to decompose T into R rank-1 tensors.
    
    def try_decompose(rank_R, T_target, max_iter=500, restarts=5):
        """
        Attempts to decompose T_target into rank_R rank-1 tensors.
        Uses a greedy approach: at each step, find the best u, v, w 
        that maximizes <T_residual, u v w^T> / (||u|| ||v|| ||w||)
        and subtracts it.
        """
        T_res = T_target.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for r in range(rank_R):
            # Find best rank-1 approximation of T_res
            # This is the NP-hard problem of best rank-1 approximation.
            # We can approximate it by iterating:
            # 1. Fix v, w, optimize u: u = T_res * v * w (element-wise product sum)
            # 2. Fix u, w, optimize v: v = T_res * u * w
            # 3. Fix u, v, optimize w: w = T_res * u * v
            
            # Initialize u, v, w randomly or from a principal component
            # Let's use a few random restarts for the inner optimization
            
            best_u = None
            best_v = None
            best_w = None
            best_score = -1
            
            for _ in range(10): # 10 random inits for the inner loop
                u = np.random.randn(dim_u)
                v = np.random.randn(dim_v)
                w = np.random.randn(dim_w)
                
                # Optimize
                for _ in range(50): # 50 iterations of alternating minimization
                    # Update u
                    # T_res * v * w means: for each i, sum_{j,k} T_res[i,j,k] v[j] w[k]
                    # This is equivalent to (T_res * v * w) where * is element-wise? No.
                    # u[i] = sum_{j,k} T_res[i,j,k] v[j] w[k]
                    # This is a matrix-vector product on the unfolded tensor.
                    # T_res is (dim_u, dim_v, dim_w)
                    # u = np.einsum('ijk,j,k->i', T_res, v, w)
                    u_new = np.einsum('ijk,j,k->i', T_res, v, w)
                    # Update v
                    v_new = np.einsum('ijk,i,k->j', T_res, u_new, w)
                    # Update w
                    w_new = np.einsum('ijk,i,j->k', T_res, u_new, v_new)
                    
                    u = u_new
                    v = v_new
                    w = w_new
                    
                    # Normalize to prevent blowup, though direction is what matters
                    # Actually, we want the specific scale to subtract correctly.
                    # The rank-1 tensor is u v w^T.
                    # The optimal scale alpha for minimizing ||T_res - alpha u v w^T||^2
                    # is alpha = <T_res, u v w^T> / ||u v w^T||^2
                    # <T_res, u v w^T> = sum_{i,j,k} T_res[i,j,k] u[i] v[j] w[k]
                    # ||u v w^T||^2 = ||u||^2 ||v||^2 ||w||^2
                    
                    dot = np.einsum('ijk,i,j,k->', T_res, u, v, w)
                    norm_sq = np.dot(u, u) * np.dot(v, v) * np.dot(w, w)
                    
                    if norm_sq < 1e-10:
                        continue
                        
                    alpha = dot / norm_sq
                    score = dot # We want to maximize the projection
                    
                    if score > best_score:
                        best_score = score
                        best_u = u.copy()
                        best_v = v.copy()
                        best_w = w.copy()
                        best_alpha = alpha
                        
                # End inner restarts
            
            if best_u is None:
                return None
                
            # Subtract the found rank-1 term
            T_res -= best_alpha * np.einsum('i,j,k->ijk', best_u, best_v, best_w)
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
            
            # Check convergence
            if np.linalg.norm(T_res) < 1e-5 * np.linalg.norm(T_target):
                # We have a decomposition!
                # But we need to ensure the entries are "integer/half-integer" or close.
                # The problem requires "integer/half-integer entries".
                # Our numerical search produces floats.
                # We need to round them and verify.
                
                U_mat = np.column_stack(U_list)
                V_mat = np.column_stack(V_list)
                W_mat = np.column_stack(W_list)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_mat * 2) / 2
                V_rounded = np.round(V_mat * 2) / 2
                W_rounded = np.round(W_mat * 2) / 2
                
                # Verify exactness
                T_recon = np.zeros((dim_u, dim_v, dim_w))
                for r in range(rank_R):
                    T_recon += np.einsum('i,j,k->ijk', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                
                if np.allclose(T_recon, T_target, atol=1e-9):
                    return (U_rounded, V_rounded, W_rounded)
                    
        return None

    # Try ranks from 30 down to 25
    for R in range(30, 24, -1):
        if time.time() > deadline - 10:
            break
            
        decomp = try_decompose(R, T)
        if decomp:
            U, V, W = decomp
            best_rank = R
            best_decomp = (U, V, W)
            break # Found a valid decomposition, stop searching lower? 
            # Actually, we might find a lower one later, but let's return the first valid one 
            # to save time, or continue if time permits. 
            # Given the difficulty, returning the first valid < 40 is a win.
            # Let's try to go lower if we have time.
            # But for safety, let's just return this one if it's < 40.
            # If we want to be aggressive, we can continue.
            # Let's stick to returning the first valid one found in this loop.

    # If no better rank found, return the schoolbook
    if best_decomp is None:
        U, V, W = U_best, V_best, W_best
    else:
        U, V, W = best_decomp

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
