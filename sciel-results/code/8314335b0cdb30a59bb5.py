import numpy as np
import time

def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Generate the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1

    def als_decompose(r, max_iter=2000, tol=1e-12, seed=0, init_U=None, init_V=None, init_W=None):
        np.random.seed(seed)
        if init_U is not None:
            U = init_U.copy()
        else:
            U = np.random.randn(n*m, r) * 0.5
        if init_V is not None:
            V = init_V.copy()
        else:
            V = np.random.randn(m*p, r) * 0.5
        if init_W is not None:
            W = init_W.copy()
        else:
            W = np.random.randn(p*n, r) * 0.5

        prev_residual = np.inf
        stagnant_count = 0

        for iteration in range(max_iter):
            if time.time() > deadline:
                break

            # Update U
            VW = (V[:, None, :] * W[None, :, :]).reshape(m*p * p*n, r)
            for a in range(n*m):
                U[a, :] = np.linalg.lstsq(VW, T[a, :, :].flatten(), rcond=None)[0]

            # Update V
            UW = (U[:, None, :] * W[None, :, :]).reshape(n*m * p*n, r)
            for b in range(m*p):
                V[b, :] = np.linalg.lstsq(UW, T[:, b, :].flatten(), rcond=None)[0]

            # Update W
            UV = (U[:, None, :] * V[None, :, :]).reshape(n*m * m*p, r)
            for c in range(p*n):
                W[c, :] = np.linalg.lstsq(UV, T[:, :, c].flatten(), rcond=None)[0]

            # Check convergence
            T_hat = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T_hat - T)

            if residual < tol:
                break

            if residual > prev_residual * 0.999:
                stagnant_count += 1
            else:
                stagnant_count = 0
            prev_residual = residual

            if stagnant_count > 100:
                break

        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T_hat - T)
        return U, V, W, residual

    # Fallback: rank 21 Strassen
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    rank_21 = 21
    U_21 = np.zeros((n*m, rank_21))
    V_21 = np.zeros((m*p, rank_21))
    W_21 = np.zeros((p*n, rank_21))

    num_blocks = p // 2
    for k in range(num_blocks):
        base = k * 7
        U_21[:, base:base+7] = U_local
        V_21[2*k, base:base+7] = V_local[0]
        V_21[2*k+1, base:base+7] = V_local[1]
        V_21[6+2*k, base:base+7] = V_local[2]
        V_21[7+2*k, base:base+7] = V_local[3]
        W_21[4*k, base:base+7] = W_local[0]
        W_21[4*k+2, base:base+7] = W_local[1]
        W_21[4*k+1, base:base+7] = W_local[2]
        W_21[4*k+3, base:base+7] = W_local[3]

    best_rank = 21
    best_U, best_V, best_W = U_21, V_21, W_21

    # Try to find a lower-rank decomposition
    # Strategy: Start from rank 21, drop one component, and refine.
    # Also try random rank 20 searches.
    
    for r in [20, 19]:
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Attempt 1: Drop a component from rank 21 and refine
        # We have 21 components. Drop the last one (index 20) and see if we can fit the rest.
        # This is a heuristic.
        if r == 20:
            U_init = U_21[:, :20]
            V_init = V_21[:, :20]
            W_init = W_21[:, :20]
            
            # Try dropping different components
            for drop_idx in range(21):
                if time.time() > deadline - 5:
                    break
                mask = [i for i in range(21) if i != drop_idx]
                U_init = U_21[:, mask]
                V_init = V_21[:, mask]
                W_init = W_21[:, mask]
                
                U, V, W, residual = als_decompose(r, max_iter=500, seed=drop_idx, init_U=U_init, init_V=V_init, init_W=W_init)
                if residual < 1e-10:
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            break
        
        # Attempt 2: Random search for rank r
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            U, V, W, residual = als_decompose(r, max_iter=500, seed=trial + 1000)
            if residual < 1e-10:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
