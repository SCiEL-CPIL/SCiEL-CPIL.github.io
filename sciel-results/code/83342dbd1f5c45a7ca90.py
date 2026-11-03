import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    N = n * m * p
    target_rank = N
    
    # Target tensor T is all ones
    T = np.ones((n, m, p))
    
    # Try to find a decomposition with rank < N
    # We iterate downwards from N-1
    for rank in range(N - 1, max(50, N - 20), -1):
        # Initialize with the first 'rank' terms of the trivial decomposition
        # Trivial terms are e_i \otimes e_j \otimes e_k
        # We need to map (i,j,k) to index t
        # t = i*m*p + j*p + k
        
        # We will use a subset of the trivial basis
        # To allow for mixing, we should initialize with random values or the trivial ones
        # Let's stick to the trivial ones for the first 'rank' indices, and 0 for others
        # This might not work well because the missing terms create a large error.
        # Better: Initialize with trivial, then set some to 0 and optimize.
        
        U_init = np.zeros((n * m, rank))
        V_init = np.zeros((m * p, rank))
        W_init = np.zeros((p * n, rank))
        
        # Select 'rank' indices from 0 to N-1
        # We will use the first 'rank' indices
        indices = np.arange(rank)
        
        for t in range(rank):
            idx = indices[t]
            i = idx // (m * p)
            rem = idx % (m * p)
            j = rem // p
            k = rem % p
            
            U_init[i * m + j, t] = 1.0
            V_init[j * p + k, t] = 1.0
            W_init[k * n + i, t] = 1.0
        
        # Flatten U, V, W for optimization
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def fun(x):
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
            
            # Compute tensor product sum
            # T_comp[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
            T_comp = np.zeros((n, m, p))
            for t in range(rank):
                # u_t is shape (n, m)
                u_t = U[:, t].reshape(n, m)
                # v_t is shape (m, p)
                v_t = V[:, t].reshape(m, p)
                # w_t is shape (p, n)
                w_t = W[:, t].reshape(p, n)
                
                # Outer product u_t \otimes v_t \otimes w_t
                # We need to add to T_comp[i,j,k] += u_t[i,j] * v_t[j,k] * w_t[k,i]
                # This is a specific contraction
                # Loop is slow, but n,m,p are small
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_comp[i, j, k] += u_t[i, j] * v_t[j, k] * w_t[k, i]
            
            # Reshape to match T
            return (T_comp - T).flatten()
        
        try:
            result = least_squares(fun, x0, method='lm', max_nfev=10000)
            if result.success and np.linalg.norm(result.fun) < 1e-5:
                # Extract solution
                U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                V_sol = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_sol = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                # Verify
                T_check = np.zeros((n, m, p))
                for t in range(rank):
                    u_t = U_sol[:, t].reshape(n, m)
                    v_t = V_sol[:, t].reshape(m, p)
                    w_t = W_sol[:, t].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_check[i, j, k] += u_t[i, j] * v_t[j, k] * w_t[k, i]
                
                if np.allclose(T_check, T, atol=1e-5):
                    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
        except Exception:
            pass

    # Fallback to trivial solution
    rank = N
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
