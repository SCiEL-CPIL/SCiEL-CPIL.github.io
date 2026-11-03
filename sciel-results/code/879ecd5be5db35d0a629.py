import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Build the target tensor T with shape (n*m, m*p, p*n) = (16, 32, 32)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a low-rank decomposition
    best_rank = n*m*p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 96 down
    for rank in range(96, 79, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(300):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Iterative update
            for it in range(3000):
                if time.time() > deadline - 15:
                    break
                
                # Update U: fix V, W
                VW = np.einsum('jr,kr->jkr', V, W).reshape(m*p * p*n, rank)
                T1 = T.reshape(n*m, m*p * p*n)
                U_new, _, _, _ = np.linalg.lstsq(VW.T, T1.T, rcond=None)
                U = U_new.T
                
                # Update V: fix U, W
                UW = np.einsum('ir,kr->ikr', U, W).reshape(n*m * p*n, rank)
                T2 = T.reshape(m*p, n*m * p*n)
                V_new, _, _, _ = np.linalg.lstsq(UW.T, T2.T, rcond=None)
                V = V_new.T
                
                # Update W: fix U, V
                UV = np.einsum('ir,jr->ijr', U, V).reshape(n*m * m*p, rank)
                T3 = T.reshape(p*n, n*m * m*p)
                W_new, _, _, _ = np.linalg.lstsq(UV.T, T3.T, rcond=None)
                W = W_new.T
                
                # Check residual every 20 iterations
                if it % 20 == 0:
                    residual = np.einsum('ir,jr,kr->ijk', U, V, W)
                    err = np.linalg.norm(residual - T)
                    if err < 1e-8:
                        found = True
                        break
            
            if found:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if found:
            break
    
    # Convert to lists
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fall back to schoolbook
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
