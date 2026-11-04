import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    I, J, K = n*m, m*p, p*n
    # Build T
    T = np.zeros((I, J, K))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    # Try rank 28
    deadline = time.time() + 140 # 10s margin
    best_result = None
    
    for rank in [28]:
        for seed in range(20): # Try 20 seeds
            if time.time() > deadline:
                break
            np.random.seed(seed)
            U = np.random.randn(I, rank) * 0.1
            V = np.random.randn(J, rank) * 0.1
            W = np.random.randn(K, rank) * 0.1
            
            # ALS
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                for u in range(I):
                    A = T[u, :, :] # J x K
                    # M = kron(W, V)
                    M = np.kron(W, V)
                    b = A.flatten()
                    # Solve M X = b
                    X, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    U[u, :] = X
                
                # Update V
                for v in range(J):
                    A = T[:, v, :] # I x K
                    M = np.kron(W, U.T)
                    b = A.flatten()
                    y, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    V[v, :] = y.T
                
                # Update W
                for w in range(K):
                    A = T[:, :, w] # I x J
                    M = np.kron(V, U)
                    b = A.flatten()
                    z, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    W[w, :] = z.T
                
                # Check residual
                # T_rec = sum_c U[:,c] V[:,c] W[:,c]
                # Efficiently:
                # Residual norm
                # Reconstructing full T is expensive (I*J*K = 900)
                # But we can check a few random entries or use norm
                # Let's compute residual norm every few iterations
                if it % 5 == 0:
                    T_rec = np.tensordot(U, V, axes=1) # I x J x rank? No.
                    # T_rec[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
                    # Use einsum
                    T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
                    res = np.linalg.norm(T_rec - T)
                    if res < 1e-8:
                        break
            
            # Round and verify
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_rec = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
            if np.allclose(T_rec, T, atol=1e-9):
                # Valid
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
    # Fallback to rank 30
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
