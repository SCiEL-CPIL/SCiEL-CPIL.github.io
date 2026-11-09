import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 3, 10
    target_rank = 60 # Start with schoolbook
    
    # We will try to find a lower rank
    # The tensor is all ones
    
    # We can try a few candidate ranks
    # Based on R(2,2,2)=7, R(2,2,3)=10, maybe R(2,3,10) is around 40-50?
    
    # Let's try to find a solution for rank r
    # We initialize U, V, W with random values
    # and minimize the error
    
    best_solution = None
    best_rank = n * m * p
    
    # We'll try ranks from 35 to 55
    # If we find a valid one, we return it
    
    # To save time, we'll only try a few random restarts for each rank
    
    for r in range(35, 56):
        # Check if we already have a better solution
        if r >= best_rank:
            continue
            
        # Try 5 random restarts
        for restart in range(5):
            np.random.seed(42 + restart)
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # We want to solve sum_r U[i,j,r] V[j,k,r] W[i,k,r] = 1
            # This is equivalent to finding a rank-r approximation of the all-ones tensor
            
            # We can use gradient descent
            # Error E = sum (1 - sum_r ...)^2
            
            # For efficiency, we can precompute some terms?
            # Or just use scipy minimize
            
            # Let's define the objective
            def objective(params):
                U_flat, V_flat, W_flat = np.split(params, [n*m*r, n*m*r + m*p*r])
                U_mat = U_flat.reshape(n*m, r)
                V_mat = V_flat.reshape(m*p, r)
                W_mat = W_flat.reshape(p*n, r)
                
                # Compute the tensor contraction
                # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # We can compute this by iterating over r
                # T = np.zeros((n, m, p))
                # for r_idx in range(r):
                #     u_col = U_mat[:, r_idx]
                #     v_col = V_mat[:, r_idx]
                #     w_col = W_mat[:, r_idx]
                #     # u_col[i*m+j], v_col[j*p+k], w_col[k*n+i]
                #     # This is not a simple outer product because indices are coupled
                #     # u[i*m+j] * v[j*p+k] * w[k*n+i]
                
                # Let's reshape to make it easier
                # U_r[i, j] = U_mat[i*m+j, r]
                # V_r[j, k] = V_mat[j*p+k, r]
                # W_r[k, i] = W_mat[k*n+i, r]
                
                # T[i, j, k] = sum_r U_r[i, j] * V_r[j, k] * W_r[k, i]
                
                # We can compute this using einsum
                # U_r: (n, m, r)
                # V_r: (m, p, r)
                # W_r: (p, n, r)
                
                U_reshaped = U_mat.reshape(n, m, r)
                V_reshaped = V_mat.reshape(m, p, r)
                W_reshaped = W_mat.reshape(p, n, r)
                
                # T[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                # This is a contraction over r, but indices i,j,k are coupled in W.
                
                # Let's just loop over r for simplicity in the objective function
                # It might be slow, but r is small (~40) and n,m,p are small.
                
                T = np.zeros((n, m, p))
                for r_idx in range(r):
                    u = U_reshaped[:, :, r_idx] # (n, m)
                    v = V_reshaped[:, :, r_idx] # (m, p)
                    w = W_reshaped[:, :, r_idx] # (p, n)
                    
                    # u[i, j] * v[j, k] * w[k, i]
                    # We need to sum over j and k? No, T[i, j, k] is the term for fixed i, j, k.
                    # Wait, the decomposition is for the tensor T.
                    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
                    
                    # So for fixed i, j, k, the term is u[i, j] * v[j, k] * w[k, i]
                    
                    # We can compute this for all i, j, k
                    # u[i, j] is constant for fixed i, j
                    # v[j, k] is constant for fixed j, k
                    # w[k, i] is constant for fixed k, i
                    
                    # T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                    
                    # We can do this with broadcasting
                    # u[i, j] -> (n, m, 1)
                    # v[j, k] -> (1, m, p)
                    # w[k, i] -> (p, 1, n) -> need to transpose to (n, p, 1)? No.
                    # w[k, i] has indices k, i. We need it at [i, j, k].
                    # So w[k, i] should be placed at index [i, :, k].
                    # w.T is (n, p). w.T[i, k] = w[k, i].
                    # So w_T = w.T -> (n, p).
                    # w_T[i, k] -> (n, 1, p)
                    
                    u_exp = u[:, :, np.newaxis] # (n, m, 1)
                    v_exp = v[np.newaxis, :, :] # (1, m, p)
                    w_T = w.T # (n, p)
                    w_exp = w_T[:, np.newaxis, :] # (n, 1, p)
                    
                    T += u_exp * v_exp * w_exp
                
                error = np.sum((1 - T)**2)
                return error
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead or Powell for small dimensions?
            # r=40 -> 40*6 + 40*30 + 40*20 = 240 + 1200 + 800 = 2240 variables.
            # This is too many for Nelder-Mead.
            
            # We need a gradient-based method.
            # We can compute the gradient analytically or use finite differences (too slow).
            
            # Let's use scipy.optimize.minimize with method='L-BFGS-B' or 'Powell'
            # But we need gradients.
            
            # Gradient of E = sum (1 - T_ijk)^2
            # dE/dU[i, j, r] = -2 sum_ijk (1 - T_ijk) * T_ijk_partial_U
            # T_ijk_partial_U[i, j, r] = V[j, k, r] * W[k, i, r] (summed over k? No, for fixed i, j, k)
            # Wait, T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            # So dT[i, j, k]/dU[i, j, r] = V[j, k, r] W[k, i, r]
            
            # We can compute the gradient efficiently.
            
            def objective_and_grad(params):
                U_flat, V_flat, W_flat = np.split(params, [n*m*r, n*m*r + m*p*r])
                U_mat = U_flat.reshape(n, m, r)
                V_mat = V_flat.reshape(m, p, r)
                W_mat = W_flat.reshape(p, n, r)
                
                T = np.zeros((n, m, p))
                for r_idx in range(r):
                    u = U_mat[:, :, r_idx]
                    v = V_mat[:, :, r_idx]
                    w = W_mat[:, :, r_idx]
                    u_exp = u[:, :, np.newaxis]
                    v_exp = v[np.newaxis, :, :]
                    w_T = w.T
                    w_exp = w_T[:, np.newaxis, :]
                    T += u_exp * v_exp * w_exp
                
                error = np.sum((1 - T)**2)
                
                # Gradient
                dU = np.zeros_like(U_mat)
                dV = np.zeros_like(V_mat)
                dW = np.zeros_like(W_mat)
                
                diff = 1 - T # (n, m, p)
                
                for r_idx in range(r):
                    u = U_mat[:, :, r_idx]
                    v = V_mat[:, :, r_idx]
                    w = W_mat[:, :, r_idx]
                    
                    # dT[i, j, k]/dU[i, j, r] = v[j, k] * w[k, i]
                    # dU[i, j, r] = -2 sum_k diff[i, j, k] * v[j, k] * w[k, i]
                    
                    # v[j, k] is (m, p)
                    # w[k, i] is (p, n) -> w.T[i, k] is (n, p)
                    
                    # sum_k diff[i, j, k] * v[j, k] * w[k, i]
                    # diff[i, j, k] * v[j, k] -> sum over k
                    # Let A[i, j, k] = diff[i, j, k] * v[j, k]
                    # Then sum_k A[i, j, k] * w[k, i]
                    
                    # A = diff * v[np.newaxis, :, :] # (n, m, p)
                    # B[i, j, i'] = sum_k A[i, j, k] * w[k, i']
                    # w is (p, n). w[k, i']
                    
                    A = diff * v[np.newaxis, :, :] # (n, m, p)
                    # B[i, j, i'] = sum_k A[i, j, k] * w[k, i']
                    # This is a tensor contraction
                    # B = np.einsum('ijk,ki->iji', A, w)
                    B = np.einsum('ijk,ki->iji', A, w)
                    dU[:, :, r_idx] = -2 * B
                    
                    # Similarly for V
                    # dT[i, j, k]/dV[j, k, r] = u[i, j] * w[k, i]
                    # dV[j, k, r] = -2 sum_i diff[i, j, k] * u[i, j] * w[k, i]
                    
                    C = diff * u[:, :, np.newaxis] # (n, m, p)
                    # sum_i C[i, j, k] * w[k, i]
                    # D[j, k, i'] = sum_i C[i, j, k] * w[k, i']
                    D = np.einsum('ijk,ki->jki', C, w)
                    dV[:, :, r_idx] = -2 * D[:, :, :] # D is (m, p, n)? No, D[j, k, i]
                    # We need dV[j, k, r]. So sum over i.
                    # D[j, k] = sum_i ...
                    # Let's re-verify einsum
                    # C[i, j, k], w[k, i]
                    # sum_i C[i, j, k] w[k, i] -> result indices j, k
                    # 'ijk,ki->jk'
                    D_jk = np.einsum('ijk,ki->jk', C, w)
                    dV[:, :, r_idx] = -2 * D_jk
                    
                    # Similarly for W
                    # dT[i, j, k]/dW[k, i, r] = u[i, j] * v[j, k]
                    # dW[k, i, r] = -2 sum_j diff[i, j, k] * u[i, j] * v[j, k]
                    
                    E = diff * u[:, :, np.newaxis] * v[np.newaxis, :, :] # (n, m, p)
                    # sum_j E[i, j, k] -> result indices i, k
                    F_ik = np.einsum('ijk->ik', E)
                    dW[:, :, r_idx] = -2 * F_ik.T # dW is (p, n). F_ik is (n, p). So F_ik.T is (p, n)
                    
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return error, grad
            
            try:
                res = minimize(objective_and_grad, params, jac=True, method='L-BFGS-B', options={'maxiter': 100, 'gtol': 1e-5})
                if res.fun < 1e-6:
                    # Check for integer/half-integer
                    U_sol = res.x[:n*m*r].reshape(n*m, r)
                    V_sol = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_sol = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Round to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Verify
                    T_check = np.zeros((n, m, p))
                    for r_idx in range(r):
                        u = U_round[:, r_idx].reshape(n, m)
                        v = V_round[:, r_idx].reshape(m, p)
                        w = W_round[:, r_idx].reshape(p, n)
                        u_exp = u[:, :, np.newaxis]
                        v_exp = v[np.newaxis, :, :]
                        w_T = w.T
                        w_exp = w_T[:, np.newaxis, :]
                        T_check += u_exp * v_exp * w_exp
                    
                    if np.allclose(T_check, 1, atol=1e-5):
                        best_solution = (U_round, V_round, W_round)
                        best_rank = r
                        break
            except Exception as e:
                pass
        
        if best_solution and best_rank == r:
            break

    if best_solution:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return schoolbook
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
