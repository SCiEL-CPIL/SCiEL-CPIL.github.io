import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Construct the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
                
    # Try to find a rank 35 decomposition
    target_rank = 35
    found = False
    best_U = None
    best_V = None
    best_W = None
    
    # Simple Gradient Descent with random restarts
    # We use a small number of iterations to stay within time limits
    for restart in range(3):
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        lr = 0.01
        for it in range(100):
            # Compute gradient
            # T_est = sum_c U_c (x) V_c (x) W_c
            # dT_est/du = V_c (x) W_c
            # dT_est/dv = U_c (x) W_c
            # dT_est/dw = U_c (x) V_c
            
            T_est = np.zeros_like(T)
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for c in range(target_rank):
                term = np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
                T_est += term
                
                # Gradients
                dU[:, c] = np.einsum('abd,b,d->a', T - T_est, V[:, c], W[:, c])
                dV[:, c] = np.einsum('abd,a,d->b', T - T_est, U[:, c], W[:, c])
                dW[:, c] = np.einsum('abd,a,b->d', T - T_est, U[:, c], V[:, c])
                
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            error = np.linalg.norm(T - T_est)
            if error < 1e-6:
                found = True
                break
        
        if found:
            break
            
    if found:
        # Round to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        T_est_rounded = np.zeros_like(T)
        for c in range(target_rank):
            T_est_rounded += np.einsum('a,b,d->abd', U_rounded[:, c], V_rounded[:, c], W_rounded[:, c])
            
        if np.linalg.norm(T - T_est_rounded) < 1e-6:
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
    # Fallback to schoolbook
    rank = rank_school
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
# EVOLVE-BLOCK-END
