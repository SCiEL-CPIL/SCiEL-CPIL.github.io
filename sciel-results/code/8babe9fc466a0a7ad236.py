import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Create the target tensor T (sparse representation)
    # T[alpha, beta, gamma] = 1 if alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    # We will store the non-zero indices and values
    nz_alpha = []
    nz_beta = []
    nz_gamma = []
    nz_val = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                nz_alpha.append(alpha)
                nz_beta.append(beta)
                nz_gamma.append(gamma)
                nz_val.append(1.0)
                
    nz_alpha = np.array(nz_alpha)
    nz_beta = np.array(nz_beta)
    nz_gamma = np.array(nz_gamma)
    nz_val = np.array(nz_val)
    
    def objective(params, rank):
        """
        Objective function for a given rank.
        params: flattened array of U, V, W
        U: (dim_u, rank)
        V: (dim_v, rank)
        W: (dim_w, rank)
        """
        # Reshape params
        u_size = dim_u * rank
        v_size = dim_v * rank
        w_size = dim_w * rank
        
        u_flat = params[:u_size]
        v_flat = params[u_size:u_size+v_size]
        w_flat = params[u_size+v_size:u_size+v_size+w_size]
        
        U = u_flat.reshape(dim_u, rank)
        V = v_flat.reshape(dim_v, rank)
        W = w_flat.reshape(dim_w, rank)
        
        # Calculate error
        # We only care about the non-zero entries of T for the "1" part,
        # but we must also penalize non-zero entries where T is 0.
        # Since the tensor is mostly zeros, we approximate the total sum of squares
        # by checking a random subset of zero-entries or using the full tensor if small enough.
        # Here 1764 is small, so we can compute the full tensor if needed, 
        # but let's stick to the non-zero part for the "1" constraint and 
        # assume the zero part is handled by the minimization if we include it.
        # Actually, ignoring the zero part is dangerous. 
        # Let's compute the full tensor $\hat{T}$ and compare to the sparse T.
        
        # Efficient computation of $\hat{T}$ at non-zero indices
        # $\hat{T}_{\alpha, \beta, \gamma} = \sum_r U_{\alpha, r} V_{\beta, r} W_{\gamma, r}$
        
        # This is a batch of dot products.
        # U[nz_alpha, :] is (N, R)
        # V[nz_beta, :] is (N, R)
        # W[nz_gamma, :] is (N, R)
        # We need sum over r.
        # prod = U[nz_alpha, :] * V[nz_beta, :] * W[nz_gamma, :]
        # err_nz = sum(prod, axis=1) - nz_val
        
        U_nz = U[nz_alpha, :]
        V_nz = V[nz_beta, :]
        W_nz = W[nz_gamma, :]
        
        prod = U_nz * V_nz * W_nz
        sum_prod = np.sum(prod, axis=1)
        
        err_nz = sum_prod - nz_val
        loss_nz = np.sum(err_nz ** 2)
        
        # Penalty for zero entries:
        # We can't easily compute all zero entries without iterating.
        # However, if the rank is low, the "leakage" to zero entries might be small.
        # To be safe, we can add a penalty on the norms of U, V, W to keep them small,
        # which tends to reduce leakage.
        # Or, we can sample a few zero entries.
        
        # Let's just use the non-zero error for now, and check validity later.
        # If the non-zero error is 0, we check if the zero errors are also 0.
        
        return loss_nz

    def check_validity(U, V, W):
        """
        Check if the decomposition is exact.
        """
        # Check non-zero entries
        U_nz = U[nz_alpha, :]
        V_nz = V[nz_beta, :]
        W_nz = W[nz_gamma, :]
        prod = U_nz * V_nz * W_nz
        sum_prod = np.sum(prod, axis=1)
        if not np.allclose(sum_prod, nz_val, atol=1e-8):
            return False
        
        # Check a sample of zero entries
        # Generate random indices not in the support
        # Support size is 42. Total 1764.
        # We can just compute the full tensor and check if it matches the sparse one.
        # Since 1764 * R is small, let's do it.
        
        # T_full shape (dim_u, dim_v, dim_w)
        # T_full[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        
        # Vectorized:
        # This is a batch of matrix multiplications?
        # No, it's an element-wise product sum.
        # We can do:
        # T_full = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        T_full = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Create sparse target in dense form
        T_target = np.zeros((dim_u, dim_v, dim_w))
        T_target[nz_alpha, nz_beta, nz_gamma] = 1.0
        
        if not np.allclose(T_full, T_target, atol=1e-8):
            return False
            
        return True

    # Try ranks from low to high
    # Schoolbook is 42. We hope for lower.
    # Start with 20, go up to 42.
    
    for rank in range(20, 43):
        # Try a few restarts
        found = False
        for restart in range(5):
            np.random.seed(42 + restart + rank)
            U_init = np.random.randn(dim_u, rank) * 0.1
            V_init = np.random.randn(dim_v, rank) * 0.1
            W_init = np.random.randn(dim_w, rank) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # We will use a simple gradient descent or scipy minimize.
            # Given the non-convexity, scipy minimize with L-BFGS-B might work.
            # But we need to provide gradients for speed.
            
            def grad(params):
                loss = objective(params, rank)
                # Finite difference gradient is too slow.
                # Let's implement analytical gradient.
                
                u_size = dim_u * rank
                v_size = dim_v * rank
                w_size = dim_w * rank
                
                u_flat = params[:u_size]
                v_flat = params[u_size:u_size+v_size]
                w_flat = params[u_size+v_size:u_size+v_size+w_size]
                
                U = u_flat.reshape(dim_u, rank)
                V = v_flat.reshape(dim_v, rank)
                W = w_flat.reshape(dim_w, rank)
                
                # Gradient of loss_nz = sum (sum_r U_a V_b W_g - 1)^2
                # dL/dU[a, r] = 2 * sum_g (sum_r' U[a, r'] V[b_g, r'] W[g_g, r'] - 1) * V[b_g, r] * W[g_g, r]
                # where b_g, g_g are the beta, gamma indices corresponding to the g-th non-zero entry.
                
                # Let's compute the residual R_g = sum_r' U[a_g, r'] V[b_g, r'] W[g_g, r'] - 1
                # Note: a_g, b_g, g_g are the indices for the g-th non-zero entry.
                # Wait, alpha is also g-indexed.
                
                U_nz = U[nz_alpha, :]
                V_nz = V[nz_beta, :]
                W_nz = W[nz_gamma, :]
                
                prod = U_nz * V_nz * W_nz
                sum_prod = np.sum(prod, axis=1)
                residual = sum_prod - nz_val
                
                # dL/dU[a, r] = 2 * sum_g residual_g * V[b_g, r] * W[g_g, r]
                # Note: for a given non-zero entry g, U[nz_alpha[g], :] is the row used.
                # So we need to scatter the gradients back to the correct rows.
                
                grad_U = np.zeros((dim_u, rank))
                grad_V = np.zeros((dim_v, rank))
                grad_W = np.zeros((dim_w, rank))
                
                # term_U_g_r = residual_g * V[b_g, r] * W[g_g, r]
                # term_V_g_r = residual_g * U[a_g, r] * W[g_g, r]
                # term_W_g_r = residual_g * U[a_g, r] * V[b_g, r]
                
                # V_nz has shape (N, R). W_nz has shape (N, R).
                # term_U = residual[:, None] * V_nz * W_nz
                term_U = residual[:, None] * V_nz * W_nz
                term_V = residual[:, None] * U_nz * W_nz
                term_W = residual[:, None] * U_nz * V_nz
                
                # Scatter add
                # np.add.at
                np.add.at(grad_U, (nz_alpha, slice(None)), 2.0 * term_U)
                np.add.at(grad_V, (nz_beta, slice(None)), 2.0 * term_V)
                np.add.at(grad_W, (nz_gamma, slice(None)), 2.0 * term_W)
                
                # Note: This gradient only accounts for the non-zero entries.
                # It ignores the zero entries. This might lead to solutions that are 
                # exact on non-zeros but have large errors on zeros.
                # However, we will check validity later.
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

            try:
                res = minimize(objective, x0, jac=grad, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.success or res.fun < 1e-8:
                    # Check validity
                    u_size = dim_u * rank
                    v_size = dim_v * rank
                    U_final = res.x[:u_size].reshape(dim_u, rank)
                    V_final = res.x[u_size:u_size+v_size].reshape(dim_v, rank)
                    W_final = res.x[u_size+v_size:u_size+v_size+w_size].reshape(dim_w, rank)
                    
                    if check_validity(U_final, V_final, W_final):
                        # Convert to list of lists as required? 
                        # The interface expects U, V, W.
                        # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
                        # The seed returns lists of lists.
                        return {
                            "U": U_final.tolist(),
                            "V": V_final.tolist(),
                            "W": W_final.tolist()
                        }
            except Exception:
                pass
                
    # Fallback to schoolbook if optimization fails
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
