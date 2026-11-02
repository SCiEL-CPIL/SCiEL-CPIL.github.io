import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 10
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    # else 0
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Flattened tensor for easier manipulation
    T_flat = T.flatten()
    indices = np.nonzero(T_flat)[0] # Not strictly needed if we loop, but good for checking
    
    def compute_residual(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Residual = sum_{u,v,w} (sum_c U[u,c] V[v,c] W[w,c] - T[u,v,w])^2
        # Efficient computation:
        # Let M_c = U_c * V_c (outer product? No, element-wise for fixed c? No)
        # For each c, we have rank 1 tensor U_c \otimes V_c \otimes W_c.
        # Sum over c.
        # Residual is squared distance to T.
        
        # We can compute the reconstructed tensor R
        R = np.zeros((U_size, V_size, W_size))
        for c in range(U.shape[1]):
            R += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.sum((R - T)**2)

    def compute_gradients(U, V, W):
        # Gradient of Residual w.r.t U, V, W
        # R = sum_c U_c \otimes V_c \otimes W_c
        # E = R - T
        # dR/dU[u,c] = V[:, c] * W[:, c] (outer product of vectors V_c and W_c)
        # dRes/dU[u,c] = 2 * sum_{v,w} E[u,v,w] V[v,c] W[w,c]
        
        R = np.zeros((U_size, V_size, W_size))
        for c in range(U.shape[1]):
            R += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        E = R - T
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(U.shape[1]):
            # dU[:, c] = 2 * sum_{v,w} E[:, v, w] V[v, c] W[w, c]
            # This is 2 * einsum('uvw,v,w->u', E, V[:, c], W[:, c])
            dU[:, c] = 2.0 * np.einsum('uvw,v,w->u', E, V[:, c], W[:, c])
            dV[:, c] = 2.0 * np.einsum('uvw,u,w->v', E, U[:, c], W[:, c])
            dW[:, c] = 2.0 * np.einsum('uvw,u,v->w', E, U[:, c], V[:, c])
            
        return dU, dV, dW

    best_solution = None
    start_time = time.time()
    max_time = 100 # Safety margin
    
    # Try multiple random restarts
    for restart in range(50):
        if time.time() - start_time > max_time:
            break
            
        np.random.seed(restart)
        # Initialize with small random values
        U = np.random.randn(U_size, target_rank) * 0.1
        V = np.random.randn(V_size, target_rank) * 0.1
        W = np.random.randn(W_size, target_rank) * 0.1
        
        # Flatten parameters
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U_f = params[:U_size * target_rank].reshape(U_size, target_rank)
            V_f = params[U_size * target_rank : U_size * target_rank + V_size * target_rank].reshape(V_size, target_rank)
            W_f = params[U_size * target_rank + V_size * target_rank :].reshape(W_size, target_rank)
            return compute_residual(U_f, V_f, W_f)
            
        def gradient(params):
            U_f = params[:U_size * target_rank].reshape(U_size, target_rank)
            V_f = params[U_size * target_rank : U_size * target_rank + V_size * target_rank].reshape(V_size, target_rank)
            W_f = params[U_size * target_rank + V_size * target_rank :].reshape(W_size, target_rank)
            dU, dV, dW = compute_gradients(U_f, V_f, W_f)
            return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

        try:
            from scipy.optimize import minimize
            res = minimize(objective, params, method='BFGS', jac=gradient, options={'maxiter': 1000, 'gtol': 1e-10})
            if res.success or res.fun < 1e-8:
                # Extract matrices
                U_opt = res.x[:U_size * target_rank].reshape(U_size, target_rank)
                V_opt = res.x[U_size * target_rank : U_size * target_rank + V_size * target_rank].reshape(V_size, target_rank)
                W_opt = res.x[U_size * target_rank + V_size * target_rank :].reshape(W_size, target_rank)
                
                # Project to half-integers
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify exactness
                if compute_residual(U_rounded, V_rounded, W_rounded) < 1e-9:
                    # Convert to lists of lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception as e:
            pass

    # Fallback to schoolbook if rank 10 not found
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
