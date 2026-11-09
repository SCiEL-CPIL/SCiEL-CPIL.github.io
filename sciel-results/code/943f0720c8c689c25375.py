import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 4, 5
    
    # Build the matrix multiplication tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose T."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        # Compute the tensor from the decomposition
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        return np.allclose(T_recon, T, atol=1e-8)
    
    # Start with the known rank-36 decomposition
    strassen_U = [
        [1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1],
        [1,1,0,0], [-1,0,1,0], [0,1,0,-1]
    ]
    strassen_V = [
        [1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0],
        [0,0,0,1], [1,1,0,0], [0,0,1,1]
    ]
    strassen_W = [
        [1,0,0,1], [0,1,0,-1], [0,0,1,1], [1,1,0,0],
        [-1,0,1,0], [0,0,0,1], [1,0,0,0]
    ]
    
    rank36 = 4 * 7 + 8
    U36 = [[0] * rank36 for _ in range(n * m)]
    V36 = [[0] * rank36 for _ in range(m * p)]
    W36 = [[0] * rank36 for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U36[li * m + (j0 + lj)][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        V36[(j0 + lj) * p + (k0 + lk)][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        W36[(k0 + lk) * n + li][col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    for i in range(n):
        for j in range(m):
            U36[i * m + j][col] = 1
            V36[j * p + 4][col] = 1
            W36[4 * n + i][col] = 1
            col += 1
    
    # Try to reduce rank by attempting rank 35 using nuclear norm relaxation
    # Strategy: use the rank-36 solution as initialization and try to find a rank-35 solution
    # by solving a convex optimization problem
    
    try:
        if time.time() > deadline - 60:
            raise Exception("No time for optimization")
        
        # Try rank 35
        target_rank = 35
        
        # Initialize with the first target_rank columns of the rank-36 solution
        U_init = np.array(U36[:, :target_rank], dtype=float)
        V_init = np.array(V36[:, :target_rank], dtype=float)
        W_init = np.array(W36[:, :target_rank], dtype=float)
        
        # Flatten for optimization
        def flatten_factors(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten_factors(x):
            U = x[:n*m*target_rank].reshape(n*m, target_rank)
            V = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            return U, V, W
        
        def objective(x):
            if time.time() > deadline - 30:
                return 1e10
            U, V, W = unflatten_factors(x)
            # Compute reconstruction error
            T_recon = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            error = np.sum((T_recon - T)**2)
            # Add nuclear norm penalty to encourage low-rank structure
            # Unfold along mode 1
            U_unfold = U  # already (n*m, target_rank)
            sv = np.linalg.svd(U_unfold, compute_uv=False)
            nuclear_norm = np.sum(sv)
            return error + 0.01 * nuclear_norm
        
        def gradient(x):
            if time.time() > deadline - 30:
                return np.zeros_like(x)
            U, V, W = unflatten_factors(x)
            T_recon = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(target_rank):
                dT = 2 * (T_recon - T)
                grad_U[:,r] = np.einsum('abc,b->a', dT, V[:,r])
                grad_V[:,r] = np.einsum('abc,a->b', dT, U[:,r])
                grad_W[:,r] = np.einsum('abc,a->c', dT, U[:,r])
            
            # Add nuclear norm gradient
            U_unfold = U
            U_s, s_vals, Vt = np.linalg.svd(U_unfold, full_matrices=False)
            grad_nuclear = U_s @ np.diag(np.ones_like(s_vals)) @ Vt  # subgradient
            grad_U += 0.01 * grad_nuclear
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        x0 = flatten_factors(U_init, V_init, W_init)
        
        # Use L-BFGS-B for optimization
        result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                         options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
        
        if result.success and result.fun < 1e-6:
            U_opt, V_opt, W_opt = unflatten_factors(result.x)
            # Check if the solution is close to integer/half-integer
            # Round to nearest half-integer
            U_rounded = np.round(2 * U_opt) / 2
            V_rounded = np.round(2 * V_opt) / 2
            W_rounded = np.round(2 * W_opt) / 2
            
            # Verify the rounded solution
            if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        # If optimization didn't work, try a different approach: 
        # Use the rank-36 solution and try to find a linear combination that gives rank 35
        # This is more complex, so we'll fall back to rank 36
        pass
        
    except Exception:
        pass
    
    # Return the known rank-36 decomposition
    return {"U": U36, "V": V36, "W": W36}
