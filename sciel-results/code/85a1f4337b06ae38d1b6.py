import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try to find rank-20 decomposition using ALS
    rank = 20
    max_rounds = 500
    
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        
        np.random.seed(attempt * 12345 + 67890)
        
        # Initialize
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Build target tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        
        # ALS: fix two, solve for third
        for rnd in range(max_rounds):
            if time.time() > deadline - 10:
                break
            
            # Solve for U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k] for all k
            # This is a linear system in U[i*m+j, :]
            # Let A_kc = V[j*p+k,c] * W[k*n+i,c], then sum_c U[i*m+j,c] * A_kc = T[i,j,k]
            # So U[i*m+j, :] = (A^T A)^{-1} A^T t where t_k = T[i,j,k]
            
            for i in range(n):
                for j in range(m):
                    # Build A matrix: (p, rank)
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    t = T[i, j, :]  # shape (p,)
                    # Solve A^T u = t in least squares
                    # u = U[i*m+j, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(A.T, t, rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Similarly for V and W
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    t = T[:, j, k]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A.T, t, rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    t = T[i, :, k]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A.T, t, rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (val - T[i,j,k])**2
            
            if err < 1e-8:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                err2 = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                            err2 += (val - T[i,j,k])**2
                
                if err2 < 1e-8:
                    U_list = [[float(x) for x in U_r[r]] for r in range(n*m)]
                    V_list = [[float(x) for x in V_r[r]] for r in range(m*p)]
                    W_list = [[float(x) for x in W_r[r]] for r in range(p*n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we got close but couldn't round, try a final polish
        if err < 1e-4:
            # Try rounding with different scales
            for scale in [1, 2, 4]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                
                err2 = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                            err2 += (val - T[i,j,k])**2
                
                if err2 < 1e-8:
                    U_list = [[float(x) for x in U_r[r]] for r in range(n*m)]
                    V_list = [[float(x) for x in V_r[r]] for r in range(m*p)]
                    W_list = [[float(x) for x in W_r[r]] for r in range(p*n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback
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
