import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def build_base_decomp():
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        strassen_decomp = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
        ]
        
        for j1 in range(2):
            for l in range(4):
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i
                        global_j = 2 * j1 + li_j
                        U[global_i * m + global_j][col] += coeff
                    
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_j = 2 * j1 + li_i
                        global_k = 2 * l + li_j
                        V[global_j * p + global_k][col] += coeff
                    
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i
                        global_k = 2 * l + li_j
                        W[global_k * n + global_i][col] += coeff
                    
                    col += 1
            
            for k in range(8):
                for j2 in range(2):
                    global_i = 2
                    global_j = 2 * j1 + j2
                    U[global_i * m + global_j][col] = 1
                    
                    global_j = 2 * j1 + j2
                    global_k = k
                    V[global_j * p + global_k][col] = 1
                    
                    global_i = 2
                    global_k = k
                    W[global_k * n + global_i][col] = 1
                    
                    col += 1
        
        return U, V, W

    U_list, V_list, W_list = build_base_decomp()
    
    U_np = np.array(U_list, dtype=np.float64)
    V_np = np.array(V_list, dtype=np.float64)
    W_np = np.array(W_list, dtype=np.float64)
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        # Vectorized verification for speed
        # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T == 1 for all i, j, k
        
        # Reshape U to (n, m, rank)
        U_r = U_mat.reshape(n, m, -1)
        V_r = V_mat.reshape(m, p, -1)
        W_r = W_mat.reshape(p, n, -1)
        
        # Compute product for all i, j, k
        # This is a batched matrix multiplication / contraction
        # Let's do it step by step to be safe and fast enough for small dims
        # Intermediate: S[j, k, i, r] ? No, let's just loop over rank if it's small, or use einsum
        
        # Using einsum is likely fastest and cleanest
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        except Exception:
            # Fallback to loops if einsum fails (unlikely)
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
        
        return np.allclose(T, 1.0, atol=tol)

    if not verify(U_np, V_np, W_np):
        # Return original if base is invalid (should not happen)
        return {"U": U_list, "V": V_list, "W": W_list}

    def try_reduce_rank(U_mat, V_mat, W_mat, max_time=25.0):
        rank = U_mat.shape[1]
        keep = [True] * rank
        start_time = time.time()
        
        # Pre-compute contributions for each rank term to speed up removal check?
        # Or just re-verify. With rank ~88 and dims 3,4,8, verification is fast.
        
        # Strategy: Greedy removal.
        # We can try to remove terms that are linearly dependent or redundant.
        
        for r in range(rank):
            if time.time() - start_time > max_time:
                break
            if not keep[r]:
                continue
            
            # Temporarily remove r
            keep[r] = False
            idx = [i for i in range(rank) if keep[i]]
            
            if len(idx) == 0:
                keep[r] = True
                continue
                
            U_red = U_mat[:, idx]
            V_red = V_mat[:, idx]
            W_red = W_mat[:, idx]
            
            if verify(U_red, V_red, W_red, tol=1e-9):
                # Successfully removed, keep it removed
                pass
            else:
                # Cannot remove
                keep[r] = True
        
        idx = [i for i in range(rank) if keep[i]]
        if len(idx) < rank:
            return U_mat[:, idx], V_mat[:, idx], W_mat[:, idx]
        return U_mat, V_mat, W_mat

    U_opt, V_opt, W_opt = try_reduce_rank(U_np, V_np, W_np, max_time=20.0)
    
    def to_list(mat):
        # Ensure entries are integer or half-integer if possible
        mat_rounded = np.round(mat * 2) / 2
        if np.allclose(mat, mat_rounded, atol=1e-9):
            return mat_rounded.tolist()
        else:
            return mat.tolist()
    
    return {"U": to_list(U_opt), "V": to_list(V_opt), "W": to_list(W_opt)}
# EVOLVE-BLOCK-END
