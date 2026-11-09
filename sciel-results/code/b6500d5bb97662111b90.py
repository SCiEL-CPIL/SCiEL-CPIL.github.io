import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def _local_search_halfint(n, m, p, r, U, V, W, time_limit=5.0):
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    candidates = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    def compute_err(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    current_err = compute_err(U, V, W)
    best_err = current_err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    while time.time() - start_time < time_limit:
        improved = False
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            choice = random.random()
            if choice < 0.33:
                row, col = random.randint(0, nm-1), random.randint(0, r-1)
                old = U[row, col]
                U[row, col] = random.choice(candidates)
            elif choice < 0.66:
                row, col = random.randint(0, mp-1), random.randint(0, r-1)
                old = V[row, col]
                V[row, col] = random.choice(candidates)
            else:
                row, col = random.randint(0, pn-1), random.randint(0, r-1)
                old = W[row, col]
                W[row, col] = random.choice(candidates)
            
            new_err = compute_err(U, V, W)
            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                improved = True
            else:
                if choice < 0.33:
                    U[row, col] = old
                elif choice < 0.66:
                    V[row, col] = old
                else:
                    W[row, col] = old
            
            if new_err < 1e-10:
                return U.copy(), V.copy(), W.copy()
        
        if not improved and best_err > 1e-10:
            if random.random() < 0.05:
                for _ in range(5):
                    c = random.random()
                    if c < 0.33:
                        best_U[random.randint(0,nm-1), random.randint(0,r-1)] = random.choice(candidates)
                    elif c < 0.66:
                        best_V[random.randint(0,mp-1), random.randint(0,r-1)] = random.choice(candidates)
                    else:
                        best_W[random.randint(0,pn-1), random.randint(0,r-1)] = random.choice(candidates)
                U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
    
    if best_err < 1e-10:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    total_deadline = time.time() + 115
    
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    
    random.seed(42)
    np.random.seed(42)
    
    for r in range(rank_schoolbook - 1, max(rank_schoolbook - 6, 0), -1):
        if time.time() > total_deadline - 5:
            break
        time_left = total_deadline - time.time() - 3
        if time_left < 1:
            break
        alloc_time = min(8.0, time_left)
        
        for start in range(3):
            if time.time() > total_deadline - 5:
                break
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            cands = np.array([-1.0, -0.5, 0.5, 1.0])
            for i in range(n * m):
                for j in range(r):
                    if random.random() < 0.4:
                        U[i, j] = random.choice(cands)
            for i in range(m * p):
                for j in range(r):
                    if random.random() < 0.4:
                        V[i, j] = random.choice(cands)
            for i in range(p * n):
                for j in range(r):
                    if random.random() < 0.4:
                        W[i, j] = random.choice(cands)
            
            if any(np.all(U[i, :] == 0) for i in range(n * m)):
                continue
            if any(np.all(V[i, :] == 0) for i in range(m * p)):
                continue
            if any(np.all(W[i, :] == 0) for i in range(p * n)):
                continue
            
            result = _local_search_halfint(n, m, p, r, U, V, W, alloc_time / 3)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
