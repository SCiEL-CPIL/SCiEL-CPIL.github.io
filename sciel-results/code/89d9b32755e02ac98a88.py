import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s limit for safety
    
    # Target ranks to try
    target_ranks = [20, 25, 30, 35, 40]
    
    best_result = None
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school

    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Try to find a decomposition of target_rank
        found = False
        for attempt in range(5): # 5 restarts per rank
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            # Scale matters, try small values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Define objective function
            def objective(params):
                U_flat = params[:n * m * target_rank]
                V_flat = params[n * m * target_rank : n * m * target_rank + m * p * target_rank]
                W_flat = params[n * m * target_rank + m * p * target_rank :]
                
                U_mat = U_flat.reshape(n * m, target_rank)
                V_mat = V_flat.reshape(m * p, target_rank)
                W_mat = W_flat.reshape(p * n, target_rank)
                
                error = 0.0
                # Check all i, j, k
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Target is 1 if (i,j,k) is a valid term
                            # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            # Sum over r should be 1
                            # Note: The tensor is sparse. Only (i,j) and (j,k) and (k,i) matter.
                            # For a fixed (i,j,k), the sum is:
                            val = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                            # The target value is 1
                            error += (val - 1.0) ** 2
                            
                            # Also need to ensure that "wrong" combinations are 0?
                            # The tensor T has entries T_{a,b,c}.
                            # a=(i,j), b=(j',k), c=(k',i').
                            # T_{a,b,c} = 1 if j=j' and i=i' and k=k'.
                            # So for a fixed i, j, k, we are checking the term where a=(i,j), b=(j,k), c=(k,i).
                            # This term MUST be 1.
                            # What about terms where b=(j',k) with j' != j?
                            # Then T_{(i,j), (j',k), (k,i)} = 0.
                            # So we must also enforce that for j' != j, the sum is 0.
                            # And for i' != i, sum is 0.
                            # And for k' != k, sum is 0.
                            
                            # This makes the objective function much more complex.
                            # We need to check all triples (i, j, k) for the "1" case.
                            # And all "0" cases.
                            
                return error

            # The objective function above only checks the "1" cases.
            # We must also check the "0" cases to ensure the tensor is exactly the multiplication tensor.
            
            def full_objective(params):
                U_flat = params[:n * m * target_rank]
                V_flat = params[n * m * target_rank : n * m * target_rank + m * p * target_rank]
                W_flat = params[n * m * target_rank + m * p * target_rank :]
                
                U_mat = U_flat.reshape(n * m, target_rank)
                V_mat = V_flat.reshape(m * p, target_rank)
                W_mat = W_flat.reshape(p * n, target_rank)
                
                error = 0.0
                # Iterate over all possible indices for U, V, W
                # U index: i in [0,n), j in [0,m) -> idx_u = i*m + j
                # V index: j' in [0,m), k in [0,p) -> idx_v = j'*p + k
                # W index: k' in [0,p), i' in [0,n) -> idx_w = k'*n + i'
                
                # The tensor T has non-zero entries only when j=j', i=i', k=k'.
                # So for each (i, j, k):
                # 1. T[(i,j), (j,k), (k,i)] should be 1.
                # 2. T[(i,j), (j',k), (k,i)] should be 0 for j' != j.
                # 3. T[(i,j), (j,k), (k',i)] should be 0 for k' != k.
                # 4. T[(i,j), (j,k), (k,i')] should be 0 for i' != i.
                
                # This is 40 * (1 + (m-1) + (p-1) + (n-1)) checks.
                # 40 * (1 + 3 + 4 + 1) = 40 * 9 = 360 checks.
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            
                            # Check 1: Should be 1
                            val = np.sum(U_mat[idx_u, :] * V_mat[idx_v, :] * W_mat[idx_w, :])
                            error += (val - 1.0) ** 2
                            
                            # Check 2: j' != j
                            for j_prime in range(m):
                                if j_prime == j: continue
                                idx_v_prime = j_prime * p + k
                                val = np.sum(U_mat[idx_u, :] * V_mat[idx_v_prime, :] * W_mat[idx_w, :])
                                error += val ** 2
                                
                            # Check 3: k' != k
                            for k_prime in range(p):
                                if k_prime == k: continue
                                idx_w_prime = k_prime * n + i
                                val = np.sum(U_mat[idx_u, :] * V_mat[idx_v, :] * W_mat[idx_w_prime, :])
                                error += val ** 2
                                
                            # Check 4: i' != i
                            for i_prime in range(n):
                                if i_prime == i: continue
                                idx_w_prime = k * n + i_prime
                                val = np.sum(U_mat[idx_u, :] * V_mat[idx_v, :] * W_mat[idx_w_prime, :])
                                error += val ** 2

                return error

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(full_objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
                
                if res.success or res.fun < 1e-9:
                    # Extract solution
                    U_sol = res.x[:n * m * target_rank].reshape(n * m, target_rank)
                    V_sol = res.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
                    W_sol = res.x[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    is_exact = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                val = np.sum(U_rounded[idx_u, :] * V_rounded[idx_v, :] * W_rounded[idx_w, :])
                                if abs(val - 1.0) > 1e-9:
                                    is_exact = False
                                    break
                                for j_prime in range(m):
                                    if j_prime == j: continue
                                    idx_v_prime = j_prime * p + k
                                    val = np.sum(U_rounded[idx_u, :] * V_rounded[idx_v_prime, :] * W_rounded[idx_w, :])
                                    if abs(val) > 1e-9:
                                        is_exact = False
                                        break
                                if not is_exact: break
                                for k_prime in range(p):
                                    if k_prime == k: continue
                                    idx_w_prime = k_prime * n + i
                                    val = np.sum(U_rounded[idx_u, :] * V_rounded[idx_v, :] * W_rounded[idx_w_prime, :])
                                    if abs(val) > 1e-9:
                                        is_exact = False
                                        break
                                if not is_exact: break
                                for i_prime in range(n):
                                    if i_prime == i: continue
                                    idx_w_prime = k * n + i_prime
                                    val = np.sum(U_rounded[idx_u, :] * V_rounded[idx_v, :] * W_rounded[idx_w_prime, :])
                                    if abs(val) > 1e-9:
                                        is_exact = False
                                        break
                                if not is_exact: break
                            if not is_exact: break
                        if not is_exact: break
                    if not is_exact: break
                    
                    if is_exact:
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = target_rank
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
