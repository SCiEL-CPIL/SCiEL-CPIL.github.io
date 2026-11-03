import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    n, m, p = 3, 3, 5
    
    # Target tensor T[I, J, K]
    # I = i*m + j (0..8)
    # J = j'*p + k (0..14)
    # K = k'*n + i' (0..14)
    # T[I, J, K] = 1 if i=i', j=j', k=k'
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    def compute_residual(U, V, W):
        R = np.einsum('ic,jc,kc->ijk', U, V, W)
        R -= T
        return R
    
    def check_exactness(U, V, W, tol=1e-4):
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if np.max(np.abs(U - U_r)) > tol or np.max(np.abs(V - V_r)) > tol or np.max(np.abs(W - W_r)) > tol:
            return False, None, None, None
            
        R = compute_residual(U_r, V_r, W_r)
        if np.max(np.abs(R)) < 1e-6:
            return True, U_r, V_r, W_r
        return False, None, None, None
    
    # Try to find a decomposition with rank 41
    # Block 1: 3x3x3 (23 terms)
    # Block 2: 3x3x2 (18 terms)
    
    rank = 41
    r1 = 23
    r2 = 18
    
    # Masks for rank 41
    mask_U = np.ones((n*m, rank))
    mask_V = np.zeros((m*p, rank))
    mask_W = np.zeros((p*n, rank))
    
    # Block 1 (0..22) corresponds to k=0,1,2
    # J indices for k=0,1,2: j*5 + k
    # j=0: 0,1,2
    # j=1: 5,6,7
    # j=2: 10,11,12
    J_block1 = [0, 1, 2, 5, 6, 7, 10, 11, 12]
    # K indices for k=0,1,2: k*3 + i
    # k=0: 0,1,2
    # k=1: 3,4,5
    # k=2: 6,7,8
    K_block1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    mask_V[J_block1, 0:r1] = 1
    mask_W[K_block1, 0:r1] = 1
    
    # Block 2 (23..40) corresponds to k=3,4
    # J indices for k=3,4: j*5 + k
    # j=0: 3,4
    # j=1: 8,9
    # j=2: 13,14
    J_block2 = [3, 4, 8, 9, 13, 14]
    # K indices for k=3,4: (k-3)*3 + i
    # k=3 (loc 0): 0,1,2 -> wait, K = k*n + i.
    # In the global tensor, K index is k*n + i'.
    # For k=3, K = 3*3 + i' = 9 + i'. So 9,10,11.
    # For k=4, K = 4*3 + i' = 12 + i'. So 12,13,14.
    K_block2 = [9, 10, 11, 12, 13, 14]
    
    mask_V[J_block2, r1:rank] = 1
    mask_W[K_block2, r1:rank] = 1
    
    # Initialize Block 2 with exact schoolbook solution
    # This ensures that part of the tensor is satisfied exactly,
    # allowing the optimizer to focus on Block 1.
    
    U_init = np.zeros((n*m, rank))
    V_init = np.zeros((m*p, rank))
    W_init = np.zeros((p*n, rank))
    
    col = r1
    for i in range(n):
        for j in range(m):
            for k_loc in range(2): # k=3,4 -> loc 0,1
                k_global = 3 + k_loc
                I = i * m + j
                J = j * p + k_global
                K = k_global * n + i
                
                U_init[I, col] = 1.0
                V_init[J, col] = 1.0
                W_init[K, col] = 1.0
                col += 1
    
    # Initialize Block 1 randomly
    np.random.seed(42)
    U_init[:, 0:r1] = np.random.randn(n*m, r1) * 0.1
    V_init[:, 0:r1] = np.random.randn(m*p, r1) * 0.1
    W_init[:, 0:r1] = np.random.randn(p*n, r1) * 0.1
    
    # Apply masks to initialization
    U_init *= mask_U
    V_init *= mask_V
    W_init *= mask_W
    
    params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def objective_and_grad(params, r, mask_U, mask_V, mask_W):
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U = params[:size_U].reshape(n*m, r)
        V = params[size_U:size_U+size_V].reshape(m*p, r)
        W = params[size_U+size_V:].reshape(p*n, r)
        
        # Apply masks
        U_masked = U * mask_U
        V_masked = V * mask_V
        W_masked = W * mask_W
        
        R = compute_residual(U_masked, V_masked, W_masked)
        loss = np.sum(R**2)
        
        grad_U = 2.0 * np.einsum('ijk,jc,kc->ic', R, V_masked, W_masked) * mask_U
        grad_V = 2.0 * np.einsum('ijk,ic,kc->jc', R, U_masked, W_masked) * mask_V
        grad_W = 2.0 * np.einsum('ijk,ic,jc->kc', R, U_masked, V_masked) * mask_W
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    best_params = params_init
    best_loss = np.inf
    
    try:
        from scipy.optimize import minimize
        
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            # Random restarts for Block 1, keep Block 2 fixed initially
            # But allow slight perturbation of Block 2 to escape local minima if needed?
            # Better to keep Block 2 exact and optimize Block 1.
            
            current_params = params_init.copy()
            # Perturb Block 1
            current_params[:n*m*r1] = U_init[:, 0:r1].flatten() + np.random.randn(n*m*r1) * 0.1
            current_params[n*m*r1:n*m*r1 + m*p*r1] = V_init[:, 0:r1].flatten() + np.random.randn(m*p*r1) * 0.1
            current_params[n*m*r1 + m*p*r1:] = W_init[:, 0:r1].flatten() + np.random.randn(p*n*r1) * 0.1
            
            # Re-apply masks
            U_cur = current_params[:n*m*r].reshape(n*m, r) * mask_U
            V_cur = current_params[n*m*r:n*m*r+m*p*r].reshape(m*p, r) * mask_V
            W_cur = current_params[n*m*r+m*p*r:].reshape(p*n, r) * mask_W
            
            current_params = np.concatenate([U_cur.flatten(), V_cur.flatten(), W_cur.flatten()])
            
            res = minimize(
                lambda x: objective_and_grad(x, rank, mask_U, mask_V, mask_W)[0],
                current_params,
                jac=lambda x: objective_and_grad(x, rank, mask_U, mask_V, mask_W)[1],
                method='L-BFGS-B',
                options={'gtol': 1e-8, 'maxiter': 1000, 'maxfun': 5000}
            )
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
            
            if best_loss < 1e-8:
                break
                
        if best_loss < 1e-8:
            U = best_params[:n*m*rank].reshape(n*m, rank)
            V = best_params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
            W = best_params[n*m*rank+m*p*rank:].reshape(p*n, rank)
            
            is_exact, U_r, V_r, W_r = check_exactness(U, V, W)
            if is_exact:
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                
    except Exception as e:
        pass
    
    # Fallback to schoolbook rank 45
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, column] = 1
                V[j*p + k, column] = 1
                W[k*n + i, column] = 1
                column += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
