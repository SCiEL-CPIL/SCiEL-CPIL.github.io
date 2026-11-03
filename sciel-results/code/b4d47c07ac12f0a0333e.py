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
    
    # Strategy: Use rank 39 = 21 (for 3x3x3 core) + 9 (for k=3) + 9 (for k=4)
    # The k=3 and k=4 slices are identity tensors of rank 9 each
    # We need to find a rank-21 decomposition for the 3x3x3 core
    
    # First, construct the fixed terms for k=3 and k=4
    # For k=3: J indices are j'*5+3 for j'=0,1,2 -> positions 3, 8, 13
    #           K indices are 3*3+i' = 9+i' for i'=0,1,2 -> positions 9, 10, 11
    # For k=4: J indices are j'*5+4 for j'=0,1,2 -> positions 4, 9, 14
    #           K indices are 4*3+i' = 12+i' for i'=0,1,2 -> positions 12, 13, 14
    
    r_core = 21
    r_extra = 18  # 9 for k=3 + 9 for k=4
    r_total = r_core + r_extra  # 39
    
    # Initialize the extra terms (k=3 and k=4)
    U_extra = np.zeros((n*m, r_extra))
    V_extra = np.zeros((m*p, r_extra))
    W_extra = np.zeros((p*n, r_extra))
    
    col = 0
    # k=3 terms: 9 terms for each (i,j)
    for i in range(n):
        for j in range(m):
            U_extra[i*m + j, col] = 1.0
            V_extra[j*p + 3, col] = 1.0  # j'*5+3 with j'=j
            W_extra[3*n + i, col] = 1.0   # k'*3+i' with k'=3, i'=i
            col += 1
    
    # k=4 terms: 9 terms for each (i,j)
    for i in range(n):
        for j in range(m):
            U_extra[i*m + j, col] = 1.0
            V_extra[j*p + 4, col] = 1.0  # j'*5+4 with j'=j
            W_extra[4*n + i, col] = 1.0   # k'*3+i' with k'=4, i'=i
            col += 1
    
    # Now optimize only the 21 core terms
    # The core terms should only have non-zero entries in:
    # U: all positions (0..8)
    # V: only J positions with k=0,1,2 (j'*5+k for k=0,1,2) -> positions 0,1,2,5,6,7,10,11,12
    # W: only K positions with k'=0,1,2 (k'*3+i' for k'=0,1,2) -> positions 0,1,2,3,4,5,6,7,8
    
    mask_V_core = np.zeros(m*p)
    for j in range(m):
        for k in range(3):  # k=0,1,2 only
            mask_V_core[j*p + k] = 1.0
    
    mask_W_core = np.zeros(p*n)
    for k in range(3):  # k'=0,1,2 only
        for i in range(n):
            mask_W_core[k*n + i] = 1.0
    
    mask_U_core = np.ones(n*m)
    
    # Target for core: T_core[I,J,K] = T[I,J,K] for J in mask_V_core support, K in mask_W_core support
    T_core = np.zeros((n*m, m*p, p*n))
    T_core[:, mask_V_core.astype(bool), :] = T[:, mask_V_core.astype(bool), :]
    # Also zero out K positions not in mask_W_core
    for k in range(p):
        if k >= 3:
            T_core[:, :, k*n:(k+1)*n] = 0.0
    
    def compute_core_residual(U_core, V_core, W_core):
        R = np.einsum('ic,jc,kc->ijk', U_core, V_core, W_core)
        R -= T_core
        return R
    
    np.random.seed(42)
    
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        
        r = r_core
        # Random initialization
        U_core = np.random.randn(n*m, r) * 0.5
        V_core = np.random.randn(m*p, r) * 0.5
        W_core = np.random.randn(p*n, r) * 0.5
        
        # Apply masks
        V_core *= mask_V_core[:, None]
        W_core *= mask_W_core[:, None]
        
        try:
            from scipy.optimize import minimize
            
            def objective_and_grad(params):
                U_c = params[:n*m*r].reshape(n*m, r)
                V_c = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_c = params[n*m*r+m*p*r:].reshape(p*n, r)
                
                V_c *= mask_V_core[:, None]
                W_c *= mask_W_core[:, None]
                
                R = compute_core_residual(U_c, V_c, W_c)
                loss = np.sum(R**2)
                
                grad_U = 2.0 * np.einsum('ijk,jc,kc->ic', R, V_c, W_c)
                grad_V = 2.0 * np.einsum('ijk,ic,kc->jc', R, U_c, W_c) * mask_V_core[:, None]
                grad_W = 2.0 * np.einsum('ijk,ic,jc->kc', R, U_c, V_c) * mask_W_core[:, None]
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return loss, grad
            
            params0 = np.concatenate([U_core.flatten(), V_core.flatten(), W_core.flatten()])
            
            res = minimize(
                lambda x: objective_and_grad(x)[0],
                params0,
                jac=lambda x: objective_and_grad(x)[1],
                method='BFGS',
                options={'gtol': 1e-10, 'maxiter': 2000}
            )
            
            if res.fun < 1e-6:
                U_c = res.x[:n*m*r].reshape(n*m, r)
                V_c = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_c = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Assemble full decomposition
                U_full = np.hstack([U_c, U_extra])
                V_full = np.hstack([V_c, V_extra])
                W_full = np.hstack([W_c, W_extra])
                
                is_exact, U_r, V_r, W_r = check_exactness(U_full, V_full, W_full)
                if is_exact:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        except Exception:
            pass
    
    # Fallback: try rank 39 with all terms optimized (no fixed structure)
    r = 39
    np.random.seed(123)
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        
        U_init = np.random.randn(n*m, r) * 0.3
        V_init = np.random.randn(m*p, r) * 0.3
        W_init = np.random.randn(p*n, r) * 0.3
        
        try:
            from scipy.optimize import minimize
            
            def obj_grad(params):
                U = params[:n*m*r].reshape(n*m, r)
                V = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W = params[n*m*r+m*p*r:].reshape(p*n, r)
                
                R = compute_residual(U, V, W)
                loss = np.sum(R**2)
                
                grad_U = 2.0 * np.einsum('ijk,jc,kc->ic', R, V, W)
                grad_V = 2.0 * np.einsum('ijk,ic,kc->jc', R, U, W)
                grad_W = 2.0 * np.einsum('ijk,ic,jc->kc', R, U, V)
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return loss, grad
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            res = minimize(
                lambda x: obj_grad(x)[0],
                params0,
                jac=lambda x: obj_grad(x)[1],
                method='BFGS',
                options={'gtol': 1e-10, 'maxiter': 1000}
            )
            
            if res.fun < 1e-6:
                U = res.x[:n*m*r].reshape(n*m, r)
                V = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                
                is_exact, U_r, V_r, W_r = check_exactness(U, V, W)
                if is_exact:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        except Exception:
            pass
    
    # Fallback to schoolbook
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
