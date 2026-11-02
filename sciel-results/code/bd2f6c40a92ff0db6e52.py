import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    def verify(U, V, W):
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        R_tensor = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d)
        return np.allclose(R_tensor, T, atol=1e-6)
        
    def try_rank(R):
        np.random.seed(42)
        # Limit time per rank
        start_time = time.time()
        max_time = 10
        
        for attempt in range(10):
            if time.time() - start_time > max_time:
                break
                
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            lr = 0.01
            for step in range(500):
                U_3d = U.reshape(n, m, R)
                V_3d = V.reshape(m, p, R)
                W_3d = W.reshape(p, n, R)
                
                R_tensor = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d)
                error = R_tensor - T
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-6:
                    if verify(U, V, W):
                        return U.tolist(), V.tolist(), W.tolist()
                    
                dU_3d = np.einsum('ijk,jkr,kr->ijr', error, V_3d, W_3d)
                dV_3d = np.einsum('ijk,ijr,kr->jkr', error, U_3d, W_3d)
                dW_3d = np.einsum('ijk,ijr,jkr->kir', error, U_3d, V_3d)
                
                U -= lr * dU_3d.reshape(n*m, R)
                V -= lr * dV_3d.reshape(m*p, R)
                W -= lr * dW_3d.reshape(p*n, R)
                
            if verify(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
        return None

    global_start = time.time()
    
    # Try ranks from 44 to 47
    for R in [44, 45, 46, 47]:
        if time.time() - global_start > 100:
            break
        res = try_rank(R)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
