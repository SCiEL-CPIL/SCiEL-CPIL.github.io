import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # Try to find a low-rank decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from low to high
    for rank in range(15, 40):
        found = False
        for seed in range(50):
            np.random.seed(seed)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating minimization
            for it in range(200):
                # Fix V, W, solve for U
                # For each (i,j): sum_r U[i*m+j, r] * (V[j*p:..., r] * W[... , r]) = 1
                # This is: for each i,j, we have a linear system in U[i*m+j, :]
                
                U_new = np.zeros((n*m, rank))
                for i in range(n):
                    for j in range(m):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                        # Let A[i*m+j, r, k] = V[j*p+k, r] * W[k*n+i, r]
                        # Then sum_r U[i*m+j, r] * A[i*m+j, r, k] = 1 for all k
                        # This is: U[i*m+j, :] @ A[i*m+j, :, :]^T = ones(p)
                        # So U[i*m+j, :] = ones(p) @ pinv(A[i*m+j, :, :]^T)
                        
                        A = np.zeros((rank, p))
                        for k in range(p):
                            for r in range(rank):
                                A[r, k] = V[j*p+k, r] * W[k*n+i, r]
                        
                        # Solve: U_vec @ A^T = ones(p)
                        # U_vec = ones(p) @ pinv(A^T)
                        try:
                            U_vec = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                        except:
                            U_vec = np.random.randn(rank)
                        U_new[i*m+j, :] = U_vec
                
                V_new = np.zeros((m*p, rank))
                for j in range(m):
                    for k in range(p):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                        A = np.zeros((rank, n))
                        for i in range(n):
                            for r in range(rank):
                                A[r, i] = U[i*m+j, r] * W[k*n+i, r]
                        try:
                            V_vec = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                        except:
                            V_vec = np.random.randn(rank)
                        V_new[j*p+k, :] = V_vec
                
                W_new = np.zeros((p*n, rank))
                for k in range(p):
                    for i in range(n):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                        A = np.zeros((rank, m))
                        for j in range(m):
                            for r in range(rank):
                                A[r, j] = U[i*m+j, r] * V[j*p+k, r]
                        try:
                            W_vec = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                        except:
                            W_vec = np.random.randn(rank)
                        W_new[k*n+i, :] = W_vec
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                            err += (val - 1)**2
                if err < 1e-10:
                    found = True
                    break
            
            if found:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank is not None:
            break
    
    if best_rank is None:
        # Fallback to naive
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
