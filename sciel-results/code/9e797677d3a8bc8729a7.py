import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    best_rank = rank_school
    best_decomp = None
    
    # Create the target tensor T
    # T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # But the tensor is defined on indices (i,j), (j',k), (i',k')
    # T[(i,j), (j',k), (i',k')] = delta(i,i') delta(j,j') delta(k,k')
    # We reshape T to (n*m, m*p, n*p)
    
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: i*m + j
                # Index in V: j*p + k
                # Index in W: k*n + i
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Function to compute tensor from U, V, W
    def tensor_from_decomp(U, V, W):
        # U: (nm, R), V: (mp, R), W: (np, R)
        # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # Efficient way:
        # T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.einsum('ir,jr,kr->ijk', U, V, W)
        
    def error(U, V, W):
        T_est = tensor_from_decomp(U, V, W)
        return np.sum((T - T_est)**2)
        
    # We will try to find a decomposition with rank R < best_rank
    # Ranges to try: 20, 22, 24, 26, 28, 30, 32, 34, 36, 38
    # We stop if we find one.
    
    def try_rank(R, max_restarts=5):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            # Scale to match the norm of T
            # T has n*m*p ones. Norm^2 = n*m*p.
            # We want U, V, W to be small.
            # Let's initialize with small random values.
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(n*p, R) * 0.1
            
            # Use gradient descent
            # We can use scipy.optimize.minimize
            # But we need to pass the gradient.
            
            # Flatten
            x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R:].reshape(n*p, R)
                return error(U_flat, V_flat, W_flat)
                
            def gradient(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R:].reshape(n*p, R)
                
                T_est = tensor_from_decomp(U_flat, V_flat, W_flat)
                diff = T - T_est
                
                # Gradient w.r.t U
                # dT/dU[i,r] = sum_{j,k} diff[i,j,k] V[j,r] W[k,r]
                # This is diff @ (V_r outer W_r)
                # More efficiently:
                # grad_U[i,r] = sum_jk diff[i,j,k] V[j,r] W[k,r]
                # = (diff * (V[:,r,None] * W[None,:,r])) .sum(axis=(1,2))?
                # Let's use einsum for clarity and correctness, then optimize if needed.
                # grad_U = np.einsum('ijk,jr,kr->ir', diff, V_flat, W_flat)
                # grad_V = np.einsum('ijk,ir,kr->jr', diff, U_flat, W_flat)
                # grad_W = np.einsum('ijk,ir,jr->kr', diff, U_flat, V_flat)
                
                grad_U = np.einsum('ijk,jr,kr->ir', diff, V_flat, W_flat)
                grad_V = np.einsum('ijk,ir,kr->jr', diff, U_flat, W_flat)
                grad_W = np.einsum('ijk,ir,jr->kr', diff, U_flat, V_flat)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                
            # Run optimizer
            # L-BFGS-B is good for this size.
            try:
                res = scipy.optimize.minimize(objective, x, jac=gradient, method='L-BFGS-B', 
                                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception as e:
                continue
                
            if res.success and res.fun < 1e-6:
                # Check if we can snap to half-integers
                U_flat = res.x[:n*m*R].reshape(n*m, R)
                V_flat = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_flat = res.x[n*m*R + m*p*R:].reshape(n*p, R)
                
                # Snap
                U_snap = np.round(U_flat * 2) / 2
                V_snap = np.round(V_flat * 2) / 2
                W_snap = np.round(W_flat * 2) / 2
                
                # Verify exactness
                T_snap = tensor_from_decomp(U_snap, V_snap, W_snap)
                if np.allclose(T, T_snap, atol=1e-9):
                    return (U_snap, V_snap, W_snap)
                else:
                    # Maybe the solution is integer only?
                    U_snap = np.round(U_flat)
                    V_snap = np.round(V_flat)
                    W_snap = np.round(W_flat)
                    T_snap = tensor_from_decomp(U_snap, V_snap, W_snap)
                    if np.allclose(T, T_snap, atol=1e-9):
                        return (U_snap, V_snap, W_snap)
                        
        return None

    # Import scipy
    import scipy.optimize
    
    # Try ranks
    # Theoretical lower bound is 20.
    # Let's try 20, 22, 24, 26, 28, 30, 32, 34, 36, 38
    for R in range(20, 40, 2):
        if time.time() > deadline:
            break
        res = try_rank(R)
        if res is not None:
            best_rank = R
            best_decomp = res
            break # Found a lower rank, stop searching for higher ranks (since we want minimal)
            
    # If no lower rank found, use schoolbook
    if best_decomp is None:
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((n*p, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_decomp = (U, V, W)
        best_rank = rank_school
        
    # Convert to lists as required by interface
    U, V, W = best_decomp
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
