import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 12, 8
    
    deadline = time.time() + 140  # Safety margin
    
    # Build the target tensor T[a,b,c]
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T[a,b,c] = 1 if i=i', j=j', k=k', else 0
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    def check_and_round(U, V, W):
        # Round to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if exact
        R = np.zeros((nm, mp, pn))
        for r in range(U_r.shape[1]):
            R += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
        
        if np.allclose(R, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    def run_als(U, V, W, max_iters=100):
        for _ in range(max_iters):
            # Fix V, W, solve for U
            # T[a] = sum_r U[a,r] * (V[:,r] * W[:,r])
            # Let M_VW = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, R)
            # T_a = T[a].reshape(-1)
            # U[a] = lstsq(M_VW, T_a)
            
            R = U.shape[1]
            M_VW = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, R)
            for a in range(nm):
                t_a = T[a, :, :].reshape(-1)
                U[a, :], _, _, _ = np.linalg.lstsq(M_VW, t_a, rcond=None)
            
            # Fix U, W, solve for V
            M_UW = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, R)
            for b in range(mp):
                t_b = T[:, b, :].reshape(-1)
                V[b, :], _, _, _ = np.linalg.lstsq(M_UW, t_b, rcond=None)
            
            # Fix U, V, solve for W
            M_UV = np.einsum('ar,br->abr', U, V).reshape(nm*mp, R)
            for c in range(pn):
                t_c = T[:, :, c].reshape(-1)
                W[c, :], _, _, _ = np.linalg.lstsq(M_UV, t_c, rcond=None)
            
            # Early stopping check
            # Calculate residual norm
            # This is expensive, so do it every 5 iterations
            # Or just rely on the final check
            
        return U, V, W

    # Try ranks from 21 to 24
    for R in [21, 22, 23, 24]:
        if time.time() > deadline - 10:
            break
            
        # Number of restarts depends on time remaining
        # Let's do 5 restarts for 21, 3 for 22, etc.
        num_restarts = 5 if R == 21 else 3
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialization
            # Use random initialization with small variance
            U = np.random.randn(nm, R) * 0.1
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1
            
            # Run ALS
            U, V, W = run_als(U, V, W, max_iters=50)
            
            # Check and round
            result = check_and_round(U, V, W)
            if result is not None:
                U_r, V_r, W_r = result
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
            # Also check without rounding in case it's exact float
            R_comp = np.zeros((nm, mp, pn))
            for r in range(R):
                R_comp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            if np.allclose(R_comp, T, atol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If R==24, we can just return schoolbook immediately to save time
        if R == 24:
             U = np.zeros((nm, R))
             V = np.zeros((mp, R))
             W = np.zeros((pn, R))
             col = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U[i*m+j, col] = 1
                         V[j*p+k, col] = 1
                         W[k*n+i, col] = 1
                         col += 1
             return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    R = n * m * p
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
