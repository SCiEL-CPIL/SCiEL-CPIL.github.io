import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    # Build target tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((num_A, num_B, num_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def residual_func(x, rank):
        U = x[:num_A * rank].reshape(num_A, rank)
        V = x[num_A * rank:num_A * rank + num_B * rank].reshape(num_B, rank)
        W = x[num_A * rank + num_B * rank:].reshape(num_C, rank)
        
        res = np.einsum('ar,br,cr->abc', U, V, W)
        return (res - T).flatten()

    # Schoolbook solution
    best_rank = n * m * p
    rank = best_rank
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to find lower rank decompositions
    for target_rank in [14, 15, 16, 17, 18, 20, 25, 30, 35]:
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(12345 + target_rank * 100 + restart)
            x0 = np.random.randn(num_A * target_rank + num_B * target_rank + num_C * target_rank) * 0.5
            
            try:
                result = least_squares(residual_func, x0, args=(target_rank,), method='trf', max_nfev=5000)
                
                if result.cost < 1e-10:
                    U_curr = result.x[:num_A * target_rank].reshape(num_A, target_rank)
                    V_curr = result.x[num_A * target_rank:num_A * target_rank + num_B * target_rank].reshape(num_B, target_rank)
                    W_curr = result.x[num_A * target_rank + num_B * target_rank:].reshape(num_C, target_rank)
                    
                    res = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
                    max_err = np.max(np.abs(res - T))
                    
                    if max_err < 1e-8:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                        found = True
                        break
                        
            except Exception:
                continue
        
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
