import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 6
    M = m * p  # 15
    P = p * n  # 10
    
    # Build target tensor
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    deadline = time.time() + 140
    best_rank = N * M * P
    best_U = None
    best_V = None
    best_W = None
    
    for rank in range(max(N, M, P), best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(150):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(attempt * 7919 + rank * 104729)
            U = np.random.randn(N, rank) * 0.5
            V = np.random.randn(M, rank) * 0.5
            W = np.random.randn(P, rank) * 0.5
            
            for iteration in range(500):
                if time.time() > deadline - 15:
                    break
                
                # Update U
                A_U = np.einsum('br,cr->bcr', V, W).reshape(M*P, rank)
                rhs_U = T.reshape(N, M*P)
                try:
                    gram = A_U.T @ A_U  # (rank, rank)
                    gram += 1e-12 * np.eye(rank)
                    rhs = A_U.T @ rhs_U.T  # (rank, N)
                    U = np.linalg.solve(gram, rhs).T  # (N, rank)
                except:
                    break
                
                # Update V
                A_V = np.einsum('ar,cr->acr', U, W).reshape(N*P, rank)
                rhs_V = T.transpose(1, 0, 2).reshape(M, N*P)
                try:
                    gram = A_V.T @ A_V
                    gram += 1e-12 * np.eye(rank)
                    rhs = A_V.T @ rhs_V.T
                    V = np.linalg.solve(gram, rhs).T
                except:
                    break
                
                # Update W
                A_W = np.einsum('ar,br->abr', U, V).reshape(N*M, rank)
                rhs_W = T.transpose(2, 0, 1).reshape(P, N*M)
                try:
                    gram = A_W.T @ A_W
                    gram += 1e-12 * np.eye(rank)
                    rhs = A_W.T @ rhs_W.T
                    W = np.linalg.solve(gram, rhs).T
                except:
                    break
            
            if time.time() > deadline - 10:
                break
            
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((recon - T) ** 2)
            
            if err < 1e-8:
                for scale in [2, 1, 4, 10]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    err_r = np.sum((recon_r - T) ** 2)
                    if err_r < 1e-10:
                        found = True
                        best_rank = rank
                        def clean(x):
                            xi = int(round(x))
                            if abs(x - xi) < 1e-9:
                                return xi
                            xh = round(x * 2) / 2
                            if abs(x - xh) < 1e-9:
                                return xh
                            return float(x)
                        best_U = [[clean(x) for x in row] for row in U_r.tolist()]
                        best_V = [[clean(x) for x in row] for row in V_r.tolist()]
                        best_W = [[clean(x) for x in row] for row in W_r.tolist()]
                        break
                if found:
                    break
        
        if found:
            break
    
    if best_U is None:
        rank = N * M * P
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_U = U
        best_V = V
        best_W = W
    
    return {"U": best_U, "V": best_V, "W": best_W}
