import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def tensor_residual(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                res = abs(s - 1.0)
                if res > max_res:
                    max_res = res
    return max_res

def round_to_half(x):
    return round(2 * x) / 2.0

def polish(U, V, W, max_iters=200, tol=1e-12):
    """Refine the decomposition to exact values using alternating least squares."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(max_iters):
        # Alternate solving for each factor
        # For U: U[i*m+j, c] = 1 / sum_k V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for c in range(rank):
                    resp = 0.0
                    for k in range(p):
                        resp += V_np[j * p + k][c] * W_np[k * n + i][c]
                    if abs(resp) > 1e-10:
                        U_np[i * m + j][c] = 1.0 / resp
                    else:
                        U_np[i * m + j][c] = 0.0
        
        # For V: V[j*p+k, c] = 1 / sum_i U[i*m+j, c] * W[k*n+i, c]
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    resp = 0.0
                    for i in range(n):
                        resp += U_np[i * m + j][c] * W_np[k * n + i][c]
                    if abs(resp) > 1e-10:
                        V_np[j * p + k][c] = 1.0 / resp
                    else:
                        V_np[j * p + k][c] = 0.0
        
        # For W: W[k*n+i, c] = 1 / sum_j U[i*m+j, c] * V[j*p+k, c]
        for k in range(p):
            for i in range(n):
                for c in range(rank):
                    resp = 0.0
                    for j in range(m):
                        resp += U_np[i * m + j][c] * V_np[j * p + k][c]
                    if abs(resp) > 1e-10:
                        W_np[k * n + i][c] = 1.0 / resp
                    else:
                        W_np[k * n + i][c] = 0.0
        
        # Check convergence
        max_res = tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist())
        if max_res < tol:
            break
    
    U_final = U_np.tolist()
    V_final = V_np.tolist()
    W_final = W_np.tolist()
    
    # Final rounding attempt
    U_rounded = [[round_to_half(x) for x in row] for row in U_final]
    V_rounded = [[round_to_half(x) for x in row] for row in V_final]
    W_rounded = [[round_to_half(x) for x in row] for row in W_final]
    
    if verify(U_rounded, V_rounded, W_rounded):
        return U_rounded, V_rounded, W_rounded
    
    return U_final, V_final, W_final

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Start with schoolbook rank 18
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try ranks from 9 up to 17
    for target_rank in range(9, 18):
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts
        n_restarts = 100
        for restart in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U_init = np.random.uniform(-1, 1, (n * m, target_rank))
            V_init = np.random.uniform(-1, 1, (m * p, target_rank))
            W_init = np.random.uniform(-1, 1, (p * n, target_rank))
            
            # Normalize
            U_init /= (np.linalg.norm(U_init) + 1e-10)
            V_init /= (np.linalg.norm(V_init) + 1e-10)
            W_init /= (np.linalg.norm(W_init) + 1e-10)
            
            U_cur, V_cur, W_cur = U_init.tolist(), V_init.tolist(), W_init.tolist()
            
            for als_iter in range(30):
                if time.time() > deadline - 30:
                    break
                
                # Run ALS
                U_cur, V_cur, W_cur = polish(U_cur, V_cur, W_cur, max_iters=10, tol=1e-8)
                
                # Check residual
                res = tensor_residual(U_cur, V_cur, W_cur)
                if res < 1e-6:
                    # Try to polish to exact
                    U_polished, V_polished, W_polished = polish(U_cur, V_cur, W_cur, max_iters=200, tol=1e-12)
                    if verify(U_polished, V_polished, W_polished):
                        best_result = {"U": U_polished, "V": V_polished, "W": W_polished}
                        best_rank = target_rank
                        break
                if time.time() > deadline - 30:
                    break
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
