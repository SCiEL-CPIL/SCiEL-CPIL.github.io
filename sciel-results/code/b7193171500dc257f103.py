import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen parameters for 2x2x2
    U_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    V_strassen = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(n * m):
                U[i][col] = U_strassen[r][i]
                
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l]][col] = V_strassen[r][l]
                
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l]][col] = W_strassen[r][l]
                
            col += 1
    
    def verify(U_mat, V_mat, W_mat):
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        
        T_expected = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n*m):
            a_row = i // m
            a_col = i % m
            for j in range(m*p):
                b_row = j // p
                b_col = j % p
                if a_col != b_row:
                    continue
                for k in range(p*n):
                    c_col = k // n
                    c_row = k % n
                    if a_row == c_row and b_col == c_col:
                        T_expected[i,j,k] = 1.0
        
        if not np.allclose(T, T_expected, atol=1e-10):
            return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Base Strassen decomposition failed verification")
    
    def try_reduce_rank(U_init, V_init, W_init, max_rank, time_limit=130):
        start_t = time.time()
        n_, m_, p_ = n, m, p
        
        U_arr = np.array(U_init, dtype=float)
        V_arr = np.array(V_init, dtype=float)
        W_arr = np.array(W_init, dtype=float)
        T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        
        def _round_and_verify(U_t, V_t, W_t, tol=1e-9):
            for scale in [1, 2]:
                U_r = np.round(U_t * scale) / scale
                V_r = np.round(V_t * scale) / scale
                W_r = np.round(W_t * scale) / scale
                T_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.allclose(T_r, T, atol=tol):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
            return None
        
        # Prioritize rank 24
        ranks_to_try = [24, 25, 26, 27]
        
        for r in ranks_to_try:
            if time.time() - start_t > time_limit:
                break
            
            best_valid = None
            # More attempts for lower ranks
            num_attempts = 100 if r == 24 else 50
            
            for attempt in range(num_attempts):
                if time.time() - start_t > time_limit:
                    break
                
                np.random.seed(1000 + attempt * 7 + r)
                if attempt < 20:
                    # Structured: drop 4 random terms from 28
                    indices = np.random.choice(28, size=r, replace=False)
                    U_try = U_arr[:, indices].copy()
                    V_try = V_arr[:, indices].copy()
                    W_try = W_arr[:, indices].copy()
                    U_try += 0.01 * np.random.randn(n*m, r)
                    V_try += 0.01 * np.random.randn(m*p, r)
                    W_try += 0.01 * np.random.randn(p*n, r)
                else:
                    U_try = np.random.randn(n*m, r) * 0.5
                    V_try = np.random.randn(m*p, r) * 0.5
                    W_try = np.random.randn(p*n, r) * 0.5
                
                max_iter = 200 if r == 24 else 100
                for iteration in range(max_iter):
                    if time.time() - start_t > time_limit:
                        break
                    
                    # Update U
                    D = np.einsum('jr,kr->jkr', V_try, W_try)
                    D_flat = D.reshape(m*p * p*n, r)
                    for i in range(n*m):
                        target = T[i, :, :].flatten()
                        U_try[i, :], _, _, _ = np.linalg.lstsq(D_flat, target, rcond=None)
                    
                    # Update V
                    D2 = np.einsum('ir,kr->ikr', U_try, W_try)
                    D2_flat = D2.reshape(n*m * p*n, r)
                    for j in range(m*p):
                        target = T[:, j, :].reshape(n*m * p*n)
                        V_try[j, :], _, _, _ = np.linalg.lstsq(D2_flat, target, rcond=None)
                    
                    # Update W
                    D3 = np.einsum('ir,jr->ijr', U_try, V_try)
                    D3_flat = D3.reshape(n*m * m*p, r)
                    for k in range(p*n):
                        target = T[:, :, k].reshape(n*m * m*p)
                        W_try[k, :], _, _, _ = np.linalg.lstsq(D3_flat, target, rcond=None)
                    
                    T_recon = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try)
                    err = np.linalg.norm(T_recon - T)
                    if err < 1e-8:
                        result = _round_and_verify(U_try, V_try, W_try)
                        if result is not None:
                            U_list, V_list, W_list = result
                            best_valid = (U_list, V_list, W_list, r)
                            break
                    if err < 1e-5 and iteration % 20 == 0:
                        result = _round_and_verify(U_try, V_try, W_try, tol=1e-6)
                        if result is not None:
                            U_list, V_list, W_list = result
                            best_valid = (U_list, V_list, W_list, r)
                            break
                
                if best_valid is not None:
                    break
            
            if best_valid is not None:
                return best_valid
        
        return None
    
    reduced = try_reduce_rank(U, V, W, max_rank=27, time_limit=135)
    
    if reduced is not None:
        U_list, V_list, W_list, r = reduced
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U, "V": V, "W": W}
