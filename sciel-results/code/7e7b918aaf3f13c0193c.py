import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Build the target tensor T
    T = np.zeros((dim_A, dim_B, dim_C), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_A = i * m + j
                idx_B = j * p + k
                idx_C = k * n + i
                T[idx_A, idx_B, idx_C] = 1

    def compute_error(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - T_calc) ** 2)

    def find_decomposition(rank, time_limit=10):
        start_time = time.time()
        U = np.random.randint(-1, 2, (dim_A, rank))
        V = np.random.randint(-1, 2, (dim_B, rank))
        W = np.random.randint(-1, 2, (dim_C, rank))
        
        current_error = compute_error(U, V, W)
        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        while time.time() - start_time < time_limit:
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
                
            i = np.random.randint(mat.shape[0])
            r = np.random.randint(rank)
            delta = np.random.choice([-1, 1])
            
            old_val = mat[i, r]
            mat[i, r] += delta
            
            new_error = compute_error(U, V, W)
            
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_error == 0:
                    return best_U, best_V, best_W
            else:
                mat[i, r] = old_val
                
        return best_U, best_V, best_W

    # Try ranks from 15 upwards
    for rank in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]:
        U, V, W = find_decomposition(rank, time_limit=5)
        if compute_error(U, V, W) == 0:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback: Trivial rank-30 decomposition
    R = 30
    U = [[0] * R for _ in range(dim_A)]
    V = [[0] * R for _ in range(dim_B)]
    W = [[0] * R for _ in range(dim_C)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_A = i * m + j
                idx_B = j * p + k
                idx_C = k * n + i
                
                U[idx_A][idx] = 1
                V[idx_B][idx] = 1
                W[idx_C][idx] = 1
                idx += 1
                
    return {
        "U": U,
        "V": V,
        "W": W
    }
