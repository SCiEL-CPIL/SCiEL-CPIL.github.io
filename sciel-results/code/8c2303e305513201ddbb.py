import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook fallback
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    start_time = time.time()
    deadline = 140
    
    for rank in range(27, rank_school):
        if time.time() - start_time > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(rank * 1000 + attempt)
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            for iteration in range(200):
                if time.time() - start_time > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.array([[V[j * p + k, c] * W[k * n + i, c] for c in range(rank)] for k in range(p)])
                        b = np.ones(p)
                        u_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i * m + j, :] = u_row
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.array([[U[i * m + j, c] * W[k * n + i, c] for c in range(rank)] for i in range(n)])
                        b = np.ones(n)
                        v_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j * p + k, :] = v_row
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.array([[U[i * m + j, c] * V[j * p + k, c] for c in range(rank)] for j in range(m)])
                        b = np.ones(m)
                        w_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k * n + i, :] = w_row
                
                # Check convergence
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            residual += (val - 1.0) ** 2
                
                if residual < 1e-10:
                    found = True
                    break
            
            if found:
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                best_rank = rank
                break
        
        if found:
            break
    
    return best_result
