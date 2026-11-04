import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin from 150s
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def check_decomposition(U, V, W, tol=1e-9):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(R, T, atol=tol):
            return False
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return True
        return False

    def find_decomposition(rank, num_attempts=100, max_iter=200):
        if time.time() > deadline:
            return None
            
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
                
            np.random.seed(42 + attempt)
            # Mix of initializations
            if attempt % 3 == 0:
                U = np.random.randn(size_u, rank)
                V = np.random.randn(size_v, rank)
                W = np.random.randn(size_w, rank)
            elif attempt % 3 == 1:
                U = np.random.choice([-1, 0, 1], size=(size_u, rank))
                V = np.random.choice([-1, 0, 1], size=(size_v, rank))
                W = np.random.choice([-1, 0, 1], size=(size_w, rank))
            else:
                U = np.random.randn(size_u, rank) * 0.1
                V = np.random.randn(size_v, rank) * 0.1
                W = np.random.randn(size_w, rank) * 0.1
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                    
                M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
                T_1 = T.reshape(size_u, size_v * size_w)
                U, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
                U = U.T
                
                M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
                T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
                V, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
                V = V.T
                
                M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
                T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
                W, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
                W = W.T
                
            if check_decomposition(U, V, W):
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    best_result = None
    # Try rank 23 first, then 24, 25, 26, 27
    for rank in range(23, 28):
        if time.time() > deadline - 10:
            break
        res = find_decomposition(rank, num_attempts=50 if rank > 23 else 200, max_iter=100 if rank > 23 else 300)
        if res is not None:
            best_result = res
            break

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
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
