import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Precompute target
    target = np.zeros((n, m, m, p, p, n))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for k in range(p):
                        for i in range(n):
                            if a == i and b == c and d == k:
                                target[a, b, c, d, k, i] = 1.0

    def compute_error(U, V, W, R):
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        total_tensor = np.zeros((n, m, m, p, p, n))
        for r in range(R):
            u_r = U_3d[:, :, r]
            v_r = V_3d[:, :, r]
            w_r = W_3d[:, :, r]
            total_tensor += np.einsum('ab,cd,ki->abcdki', u_r, v_r, w_r)
            
        return np.sum((total_tensor - target) ** 2)

    def try_find_decomposition(R):
        np.random.seed(int(time.time() * 1000) % 10000)
        for restart in range(20):
            U = np.random.choice([-1, 0, 1], size=(dim_U, R))
            V = np.random.choice([-1, 0, 1], size=(dim_V, R))
            W = np.random.choice([-1, 0, 1], size=(dim_W, R))
            
            current_err = compute_error(U, V, W, R)
            
            for _ in range(500):
                if current_err < 1e-9:
                    return U, V, W
                
                mat = np.random.choice([0, 1, 2])
                if mat == 0:
                    row = np.random.randint(dim_U)
                    col = np.random.randint(R)
                    old_val = U[row, col]
                    for delta in [-1, 1, -0.5, 0.5]:
                        new_val = old_val + delta
                        U[row, col] = new_val
                        new_err = compute_error(U, V, W, R)
                        if new_err < current_err:
                            current_err = new_err
                            break
                        else:
                            U[row, col] = old_val
                elif mat == 1:
                    row = np.random.randint(dim_V)
                    col = np.random.randint(R)
                    old_val = V[row, col]
                    for delta in [-1, 1, -0.5, 0.5]:
                        new_val = old_val + delta
                        V[row, col] = new_val
                        new_err = compute_error(U, V, W, R)
                        if new_err < current_err:
                            current_err = new_err
                            break
                        else:
                            V[row, col] = old_val
                else:
                    row = np.random.randint(dim_W)
                    col = np.random.randint(R)
                    old_val = W[row, col]
                    for delta in [-1, 1, -0.5, 0.5]:
                        new_val = old_val + delta
                        W[row, col] = new_val
                        new_err = compute_error(U, V, W, R)
                        if new_err < current_err:
                            current_err = new_err
                            break
                        else:
                            W[row, col] = old_val
                            
        return None

    # Try ranks from 23 down
    for R in range(23, 10, -1):
        if time.time() > deadline:
            break
        res = try_find_decomposition(R)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
