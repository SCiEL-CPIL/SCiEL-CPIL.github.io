import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = start_time + 110
    
    def check_validity(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)

    def is_half_integer(mat):
        two_mat = 2.0 * mat
        return np.allclose(two_mat, np.round(two_mat), atol=1e-9)

    def compute_error(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.sum((T - 1.0) ** 2)

    def local_search(U, V, W, max_iters=1000):
        current_error = compute_error(U, V, W)
        if current_error < 1e-9:
            return U, V, W, current_error
            
        candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random matrix
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
                rows = nm
            elif mat_idx == 1:
                mat = V
                rows = mp
            else:
                mat = W
                rows = pn
                
            r = mat.shape[1]
            row = random.randint(0, rows - 1)
            col = random.randint(0, r - 1)
            
            old_val = mat[row, col]
            new_val = random.choice(candidates)
            if new_val == old_val:
                continue
                
            mat[row, col] = new_val
            new_error = compute_error(U, V, W)
            
            if new_error < current_error - 1e-15:
                current_error = new_error
            else:
                mat[row, col] = old_val
                
            if current_error < 1e-9:
                break
                
        return U, V, W, current_error

    # Search for low rank
    # Try ranks from 10 to 60
    for r in range(10, 61):
        if time.time() > deadline - 5:
            break
            
        # Try a few random restarts
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randint(-4, 5, size=(nm, r)) * 0.5
            V = np.random.randint(-4, 5, size=(mp, r)) * 0.5
            W = np.random.randint(-4, 5, size=(pn, r)) * 0.5
            
            U, V, W, err = local_search(U, V, W, max_iters=500)
            
            if err < 1e-9:
                if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
