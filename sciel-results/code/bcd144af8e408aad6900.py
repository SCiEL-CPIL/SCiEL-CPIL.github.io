import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Standard rank
    rank_std = n * m * p
    U_std = [[0] * rank_std for _ in range(n * m)]
    V_std = [[0] * rank_std for _ in range(m * p)]
    W_std = [[0] * rank_std for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_std[i * m + j][col] = 1
                V_std[j * p + k][col] = 1
                W_std[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank
    # Target tensor T
    # Indices: 
    # U: i*m + j (6)
    # V: j*p + k (9)
    # W: k*n + i (6)
    # Tensor component T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Note: The sum is over j for the multiplication.
    # C_{i', k'} = sum_j A_{i', j} B_{j, k'}
    # So T_{(i', j), (j, k'), (k', i')} = 1.
    # In our indexing:
    # a = i' * m + j
    # b = j * p + k'
    # c = k' * n + i'
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.flatten()
    
    best_result = None
    best_score = float('inf')
    
    # Try rank 17
    target_rank = 17
    if target_rank < rank_std:
        start_time = time.time()
        deadline = 110 # Safety margin
        
        # Objective function
        def error_and_grad(params):
            # params: flattened U, V, W
            # U: (dim_u, r), V: (dim_v, r), W: (dim_w, r)
            U = params[0:dim_u*target_rank].reshape((dim_u, target_rank))
            V = params[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
            W = params[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            
            # Reconstruct tensor
            # T_rec = sum_t U[:,t] outer V[:,t] outer W[:,t]
            # Efficient computation:
            # T_rec[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
            # This is like a tensor product.
            
            # For gradient:
            # E = T - T_rec
            # dE/dU[a,t] = - sum_{b,c} E[a,b,c] V[b,t] W[c,t]
            # dE/dV[b,t] = - sum_{a,c} E[a,b,c] U[a,t] W[c,t]
            # dE/dW[c,t] = - sum_{a,b} E[a,b,c] U[a,t] V[b,t]
            
            # Compute T_rec
            # Using einsum: 'at,bt,ct->abc'
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            E = T_flat - T_rec.flatten()
            error = np.dot(E, E)
            
            if error < 1e-12:
                # Check if we can stop early? 
                # We need the gradient to be 0 too, but if error is 0, we found it.
                # We can return error and a dummy gradient
                grad = np.zeros_like(params)
                return error, grad
            
            E_tensor = E.reshape((dim_u, dim_v, dim_w))
            
            grad_U = -2 * np.einsum('abc,bt,ct->at', E_tensor, V, W)
            grad_V = -2 * np.einsum('abc,at,ct->bt', E_tensor, U, W)
            grad_W = -2 * np.einsum('abc,at,bt->ct', E_tensor, U, V)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return error, grad
            
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(restart)
            params = np.random.randn(dim_u*target_rank + dim_v*target_rank + dim_w*target_rank) * 0.1
            
            res = minimize(error_and_grad, params, jac=True, method='BFGS', 
                           options={'maxiter': 200, 'gtol': 1e-8, 'disp': False})
            
            if res.success or res.fun < 1e-9:
                # Check if entries are close to integer/half-integer
                U_sol = res.x[0:dim_u*target_rank].reshape((dim_u, target_rank))
                V_sol = res.x[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
                W_sol = res.x[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
                
                # Check 2*entries are close to integers
                is_half_int = True
                for mat in [U_sol, V_sol, W_sol]:
                    doubled = 2.0 * mat
                    rounded = np.round(doubled)
                    if np.max(np.abs(doubled - rounded)) > 1e-5:
                        is_half_int = False
                        break
                
                if is_half_int:
                    # Construct the result with rounded values
                    U_res = np.round(2.0 * U_sol) / 2.0
                    V_res = np.round(2.0 * V_sol) / 2.0
                    W_res = np.round(2.0 * W_sol) / 2.0
                    
                    # Verify exactness
                    T_check = np.einsum('at,bt,ct->abc', U_res, V_res, W_res)
                    if np.allclose(T_check, T, atol=1e-9):
                        # Convert to list of lists
                        U_list = U_res.tolist()
                        V_list = V_res.tolist()
                        W_list = W_res.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
        # If not found, try rank 16? Unlikely.
        
    # Return standard rank 18
    return {"U": U_std, "V": V_std, "W": W_std}
# EVOLVE-BLOCK-END
