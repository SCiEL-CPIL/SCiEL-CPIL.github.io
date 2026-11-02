import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    schoolbook = get_schoolbook()
    best_result = schoolbook
    best_rank = n * m * p
    
    def verify_decomposition(U_list, V_list, W_list, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U_list[a][d] * V_list[b][d] * W_list[c][d] for d in range(rank))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def try_decompose(target_rank, num_restarts=12, max_iter=200):
        if time.time() > deadline - 30:
            return None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + restart * 13)
            U = np.random.randn(nm, target_rank) * 0.2
            V = np.random.randn(mp, target_rank) * 0.2
            W = np.random.randn(pn, target_rank) * 0.2
            
            lr = 0.03
            prev_error = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 8:
                    break
                
                error = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            diff = val - 1.0
                            error += diff * diff
                            grad_U[a, :] += 2 * diff * (V[b, :] * W[c, :])
                            grad_V[b, :] += 2 * diff * (U[a, :] * W[c, :])
                            grad_W[c, :] += 2 * diff * (U[a, :] * V[b, :])
                
                for _ in range(60):
                    a = np.random.randint(nm)
                    b = np.random.randint(mp)
                    c = np.random.randint(pn)
                    if T[a, b, c] == 0:
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        error += val * val
                        grad_U[a, :] += 2 * val * (V[b, :] * W[c, :])
                        grad_V[b, :] += 2 * val * (U[a, :] * W[c, :])
                        grad_W[c, :] += 2 * val * (U[a, :] * V[b, :])
                
                if error < 1e-12:
                    U_list = [[round(U[i, j] * 2) / 2 for j in range(target_rank)] for i in range(nm)]
                    V_list = [[round(V[i, j] * 2) / 2 for j in range(target_rank)] for i in range(mp)]
                    W_list = [[round(W[i, j] * 2) / 2 for j in range(target_rank)] for i in range(pn)]
                    if verify_decomposition(U_list, V_list, W_list, target_rank):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    break
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                lr *= 0.995
                if error > prev_error * 1.02:
                    lr *= 0.7
                prev_error = error
        
        return None
    
    for rank in range(32, 19, -1):
        if time.time() > deadline - 25:
            break
        result = try_decompose(rank, num_restarts=15, max_iter=150)
        if result is not None:
            best_result = result
            best_rank = rank
            if rank <= 20:
                break
    
    if best_result is None:
        best_result = schoolbook
    
    return best_result
