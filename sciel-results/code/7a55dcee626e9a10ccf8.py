import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the matmul tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Try to find lower-rank decomposition via ALS
    for target_rank in [41, 40, 39]:
        if time.time() - start_time > 120:
            break
        
        np.random.seed(42)
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        for iteration in range(20):
            if time.time() - start_time > 115:
                break
            
            # Update U
            A = np.zeros((m*p*p*n, target_rank))
            for r in range(target_rank):
                A[:, r] = np.kron(V[:, r], W[:, r])
            U_pinv = np.linalg.pinv(A)
            U_new = np.zeros((n*m, target_rank))
            for i in range(n*m):
                U_new[i, :] = U_pinv @ T[i, :, :].flatten()
            U = U_new
            
            # Update V
            A_V = np.zeros((n*m*p*n, target_rank))
            for r in range(target_rank):
                A_V[:, r] = np.kron(U[:, r], W[:, r])
            V_pinv = np.linalg.pinv(A_V)
            V_new = np.zeros((m*p, target_rank))
            for j in range(m*p):
                V_new[j, :] = V_pinv @ T[:, j, :].flatten()
            V = V_new
            
            # Update W
            A_W = np.zeros((n*m*m*p, target_rank))
            for r in range(target_rank):
                A_W[:, r] = np.kron(U[:, r], V[:, r])
            W_pinv = np.linalg.pinv(A_W)
            W_new = np.zeros((p*n, target_rank))
            for k in range(p*n):
                W_new[k, :] = W_pinv @ T[:, :, k].flatten()
            W = W_new
        
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
        error = np.linalg.norm(T_rec - T)
        
        if error < 1e-6:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            T_rec_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.linalg.norm(T_rec_r - T) < 1e-6:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback: rank 42 Strassen
    U = np.zeros((n * m, 42), dtype=float)
    V = np.zeros((m * p, 42), dtype=float)
    W = np.zeros((p * n, 42), dtype=float)
    
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    col = 0
    for block in blocks:
        a_rows, a_cols, b_rows, b_cols = block
        a_map = {0: a_rows[0]*m+a_cols[0], 1: a_rows[0]*m+a_cols[1], 2: a_rows[1]*m+a_cols[0], 3: a_rows[1]*m+a_cols[1]}
        b_map = {0: b_rows[0]*p+b_cols[0], 1: b_rows[0]*p+b_cols[1], 2: b_rows[1]*p+b_cols[0], 3: b_rows[1]*p+b_cols[1]}
        w_map = {0: b_cols[0]*n+a_rows[0], 1: b_cols[1]*n+a_rows[0], 2: b_cols[0]*n+a_rows[1], 3: b_cols[1]*n+a_rows[1]}
        
        for u_c, v_c, w_c in strassen_terms:
            for li, c in u_c:
                U[a_map[li], col] += c
            for li, c in v_c:
                V[b_map[li], col] += c
            for li, c in w_c:
                W[w_map[li], col] += c
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
