import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    def check_exact(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try to find rank 47 decomposition using ALS
    for r in [47]:
        for trial in range(10):
            U = np.random.uniform(0.25, 0.75, (n*m, r))
            V = np.random.uniform(0.25, 0.75, (m*p, r))
            W = np.random.uniform(0.25, 0.75, (p*n, r))
            
            for _ in range(200):
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            for t in range(r):
                                A[k, t] = V[j*p+k, t] * W[k*n+i, t]
                        b = np.ones(p)
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            for t in range(r):
                                A[i, t] = U[i*m+j, t] * W[k*n+i, t]
                        b = np.ones(n)
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            for t in range(r):
                                A[j, t] = U[i*m+j, t] * V[j*p+k, t]
                        b = np.ones(m)
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if check_exact(U_rounded, V_rounded, W_rounded):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
    
    # Fallback to schoolbook rank 48
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
