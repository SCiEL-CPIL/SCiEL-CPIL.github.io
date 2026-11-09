import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    R = U.shape[1]
    # Reshape for einsum
    # U: (n*m, R) -> (n, m, R)
    # V: (m*p, R) -> (m, p, R)
    # W: (p*n, R) -> (p, n, R)
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    U_t = U.reshape(n, m, R)
    V_t = V.reshape(m, p, R)
    W_t = W.reshape(p, n, R)
    
    # Compute T
    # U_t[i,j,c], V_t[j,k,c], W_t[k,i,c]
    # Note: W_t is indexed by [k, i, c]
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    # Check if T is all 1s
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _try_find_rank(n, m, p, R, time_limit=5.0):
    """
    Try to find a rank-R decomposition using random restarts and local optimization.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    n_u, n_v, n_w = n*m, m*p, p*n
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        U = np.random.randn(n, m, R) * 0.5
        V = np.random.randn(m, p, R) * 0.5
        W = np.random.randn(p, n, R) * 0.5
        
        lr = 0.05
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
            
            # Compute T
            T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            Err = T - 1.0
            
            # Compute Gradients
            # dU[i,j,c] = sum_k Err[i,j,k] V[j,k,c] W[k,i,c]
            dU = np.einsum('ijk,jkc,kic->ijc', Err, V, W)
            # dV[j,k,c] = sum_i Err[i,j,k] U[i,j,c] W[k,i,c]
            dV = np.einsum('ijk,ijc,kic->jkc', Err, U, W)
            # dW[k,i,c] = sum_j Err[i,j,k] U[i,j,c] V[j,k,c]
            dW = np.einsum('ijk,ijc,jkc->kic', Err, U, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodic check
            if step % 20 == 0:
                # Round to half-integers
                U_r = np.round(2 * U) / 2.0
                V_r = np.round(2 * V) / 2.0
                W_r = np.round(2 * W) / 2.0
                
                if _verify(U_r.flatten(), V_r.flatten(), W_r.flatten(), n, m, p):
                    return U_r.flatten(), V_r.flatten(), W_r.flatten()

    return None

def construct():
    n, m, p = 2, 3, 4
    deadline = time.time() + 100.0
    
    # Try to find a lower rank decomposition
    for R in range(18, 13, -1):
        if time.time() > deadline:
            break
        res = _try_find_rank(n, m, p, R, time_limit=5.0)
        if res is not None:
            U, V, W = res
            U_out = [[float(U[i, c]) for c in range(R)] for i in range(n*m)]
            V_out = [[float(V[i, c]) for c in range(R)] for i in range(m*p)]
            W_out = [[float(W[i, c]) for c in range(R)] for i in range(p*n)]
            return {"U": U_out, "V": V_out, "W": W_out}
            
    # Fallback to schoolbook rank 24
    R = 24
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    U_out = [[float(U[i, c]) for c in range(R)] for i in range(n*m)]
    V_out = [[float(V[i, c]) for c in range(R)] for i in range(m*p)]
    W_out = [[float(W[i, c]) for c in range(R)] for i in range(p*n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
