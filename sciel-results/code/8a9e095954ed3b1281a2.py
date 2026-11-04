import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    total_elements = n * m * p
    
    def schoolbook():
        rank = total_elements
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
    
    start_time = time.time()
    time_limit = 110  # leave 10s buffer
    
    for rank in range(19, total_elements):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(rank * 1000 + attempt)
            U = np.random.randn(n * m, rank) * 0.3
            V = np.random.randn(m * p, rank) * 0.3
            W = np.random.randn(p * n, rank) * 0.3
            
            lr = 0.05
            
            for step in range(500):
                if time.time() - start_time > time_limit:
                    break
                
                err = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ui = i * m + j
                            vj = j * p + k
                            wk = k * n + i
                            
                            val = np.dot(U[ui], V[vj] * W[wk])
                            diff = val - 1.0
                            err += diff * diff
                            
                            dU[ui] += 2 * diff * (V[vj] * W[wk])
                            dV[vj] += 2 * diff * (U[ui] * W[wk])
                            dW[wk] += 2 * diff * (U[ui] * V[vj])
                
                if err < 1e-10:
                    found = True
                    break
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if found:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return schoolbook()
