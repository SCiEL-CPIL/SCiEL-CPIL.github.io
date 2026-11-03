import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    time_limit = 110  # Leave some margin
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find a lower rank decomposition using ALS
    for target_rank in range(55, 40, -1):
        if time.time() - start_time > time_limit:
            break
        found = False
        for seed in range(3):
            if time.time() - start_time > time_limit:
                break
            np.random.seed(seed)
            U_try = np.random.randn(n * m, target_rank)
            V_try = np.random.randn(m * p, target_rank)
            W_try = np.random.randn(p * n, target_rank)
            
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A_mat = np.array([[V_try[j * p + k, r] * W_try[k * n + i, r] for r in range(target_rank)] for k in range(p)])
                        try:
                            U_try[i * m + j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A_mat = np.array([[U_try[i * m + j, r] * W_try[k * n + i, r] for r in range(target_rank)] for i in range(n)])
                        try:
                            V_try[j * p + k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A_mat = np.array([[U_try[i * m + j, r] * V_try[j * p + k, r] for r in range(target_rank)] for j in range(m)])
                        try:
                            W_try[k * n + i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check error
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum([U_try[i*m+j, r] * V_try[j*p+k, r] * W_try[k*n+i, r] for r in range(target_rank)])
                            err += (val - 1) ** 2
                if err < 1e-8:
                    found = True
                    break
            
            if found:
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                break
        if found:
            break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
