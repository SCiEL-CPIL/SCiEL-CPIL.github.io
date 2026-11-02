import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Build target tensor T
    # T is a 3D array of size (n*m, m*p, p*n)
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    T = np.zeros((U_dim, V_dim, W_dim))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try ranks from 32 to 45
    for rank in range(32, 46):
        if time.time() > deadline:
            break
            
        success = False
        # Try multiple random seeds
        for seed in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(seed)
            # Initialize U, V, W
            # Scale initialization to help convergence
            U = np.random.randn(U_dim, rank) * 0.1
            V = np.random.randn(V_dim, rank) * 0.1
            W = np.random.randn(W_dim, rank) * 0.1
            
            # Optimize
            # We want to minimize || T - tensor(U,V,W) ||_F^2
            # This is a non-convex problem.
            # We can use scipy.optimize.minimize on the flattened vector.
            
            def objective(params):
                U_flat, V_flat, W_flat = params[:U_dim*rank], params[U_dim*rank:U_dim*rank+V_dim*rank], params[U_dim*rank+V_dim*rank:]
                U_c = U_flat.reshape(U_dim, rank)
                V_c = V_flat.reshape(V_dim, rank)
                W_c = W_flat.reshape(W_dim, rank)
                
                # Compute decomposition
                # D[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
                # This is expensive to compute fully in Python loop.
                # Use einsum or matmul tricks?
                # T_decomp = np.einsum('uc,vc,wc->uvw', U_c, V_c, W_c)
                D = np.einsum('uc,vc,wc->uvw', U_c, V_c, W_c)
                err = np.sum((T - D) ** 2)
                return err

            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, params, method='BFGS', options={'maxiter': 100, 'gtol': 1e-8})
                if res.fun < 1e-4:
                    # Check rounding
                    U_final = res.x[:U_dim*rank].reshape(U_dim, rank)
                    V_final = res.x[U_dim*rank:U_dim*rank+V_dim*rank].reshape(V_dim, rank)
                    W_final = res.x[U_dim*rank+V_dim*rank:].reshape(W_dim, rank)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_final * 2) / 2
                    V_round = np.round(V_final * 2) / 2
                    W_round = np.round(W_final * 2) / 2
                    
                    # Verify
                    D_check = np.einsum('uc,vc,wc->uvw', U_round, V_round, W_round)
                    if np.allclose(T, D_check, atol=1e-6):
                        # Convert to list of lists
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
            except Exception:
                pass
            
    # Fallback: Schoolbook
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
