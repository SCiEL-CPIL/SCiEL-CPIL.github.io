import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 130.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Split m=4 into 2 blocks of 2.
    # For each block (2 of them):
    #   Split n=3 into 2 (top) and 1 (bot).
    #   Top: 2x2x8. Split p=8 into 4 blocks of 2.
    #         Each 2x2x2 uses Strassen (rank 7).
    #         Total Top Rank = 4 * 7 = 28.
    #   Bot: 1x2x8. Schoolbook rank = 1*2*8 = 16.
    #   Total per m-block = 28 + 16 = 44.
    # Total Rank = 2 * 44 = 88.
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the linear combinations for U (A), V (B), W (C)
    # Indices in 2x2 block:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ({0: 1, 2: 1}, {0: 1, 3: 1}, {0: 1, 3: -1}), # Wait, W coeffs are for reconstruction
        # Let's use the standard reconstruction coefficients
    ]
    
    # Correct Strassen Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 1: M1
    # U: A00 + A11 -> {0: 1, 3: 1} (using 0:00, 1:01, 2:10, 3:11)
    # V: B00 + B11 -> {0: 1, 3: 1}
    # W: C00 + C11 -> {0: 1, 3: 1}
    
    # Term 2: M2
    # U: A10 + A11 -> {2: 1, 3: 1}
    # V: B00 -> {0: 1}
    # W: C10 - C11 -> {2: 1, 3: -1}
    
    # Term 3: M3
    # U: A00 -> {0: 1}
    # V: B01 - B11 -> {1: 1, 3: -1}
    # W: C01 + C11 -> {1: 1, 3: 1}
    
    # Term 4: M4
    # U: A11 -> {3: 1}
    # V: B10 - B00 -> {2: 1, 0: -1}
    # W: C00 + C10 -> {0: 1, 2: 1}
    
    # Term 5: M5
    # U: A00 + A01 -> {0: 1, 1: 1}
    # V: B11 -> {3: 1}
    # W: C01 - C00 -> {1: 1, 0: -1}
    
    # Term 6: M6
    # U: A10 - A00 -> {2: 1, 0: -1}
    # V: B00 + B01 -> {0: 1, 1: 1}
    # W: C11 -> {3: 1}
    
    # Term 7: M7
    # U: A01 - A11 -> {1: 1, 3: -1}
    # V: B10 + B11 -> {2: 1, 3: 1}
    # W: C00 -> {0: 1}
    
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
        # Top part: 2x2x8
        for l in range(4):
            for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                # Map local indices to global
                # A (U)
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_j = 2 * j1 + li_j
                    U[global_i * m + global_j][col] += coeff
                
                # B (V)
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_j = 2 * j1 + li_i
                    global_k = 2 * l + li_j
                    V[global_j * p + global_k][col] += coeff
                
                # C (W)
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_k = 2 * l + li_j
                    W[global_k * n + global_i][col] += coeff
                
                col += 1
        
        # Bot part: 1x2x8
        # Rank 16
        for k in range(8):
            for j2 in range(2):
                # Term: A[2, 2*j1+j2] * B[2*j1+j2, k] * C[2, k]
                
                # U
                global_i = 2
                global_j = 2 * j1 + j2
                U[global_i * m + global_j][col] = 1
                
                # V
                global_j = 2 * j1 + j2
                global_k = k
                V[global_j * p + global_k][col] = 1
                
                # W
                global_i = 2
                global_k = k
                W[global_k * n + global_i][col] = 1
                
                col += 1

    # Convert to numpy for efficient checking
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Fast vectorized verification
    def verify_fast(U_mat, V_mat, W_mat):
        """Vectorized check: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
        n, m, p = DIMENSIONS
        # Precompute index arrays
        idx_u = np.array([i*m+j for i in range(n) for j in range(m)])  # (n*m,)
        idx_v = np.array([j*p+k for j in range(m) for k in range(p)])  # (m*p,)
        idx_w = np.array([k*n+i for k in range(p) for i in range(n)])  # (p*n,)
        
        # Reshape for broadcasting: we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Shape: (n, m, p, rank)
        rank = U_mat.shape[1]
        # U[i*m+j, r] -> (n, m, 1, rank)
        U_reshaped = U_mat[idx_u].reshape(n, m, 1, rank)
        # V[j*p+k, r] -> (1, m, p, rank)
        V_reshaped = V_mat[idx_v].reshape(1, m, p, rank)
        # W[k*n+i, r] -> (n, 1, p, rank)
        W_reshaped = W_mat[idx_w].reshape(n, 1, p, rank)
        
        # Product and sum over rank
        result = np.sum(U_reshaped * V_reshaped * W_reshaped, axis=3)
        return np.allclose(result, 1.0, atol=1e-9)
    
    if not verify_fast(U_np, V_np, W_np):
        return {"U": U, "V": V, "W": W}
    
    # Aggressive rank reduction with vectorized verification
    def try_reduce_rank(U_mat, V_mat, W_mat, max_time=60.0):
        """Try to reduce rank by greedily removing columns, with multi-pass."""
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        keep = np.ones(rank, dtype=bool)
        start_time = time.time()
        
        # Precompute for fast verification
        idx_u = np.array([i*m+j for i in range(n) for j in range(m)])
        idx_v = np.array([j*p+k for j in range(m) for k in range(p)])
        idx_w = np.array([k*n+i for k in range(p) for i in range(n)])
        
        def check_valid(keep_mask):
            """Check if decomposition holds with given columns kept."""
            idx = np.where(keep_mask)[0]
            if len(idx) == 0:
                return False
            U_r = U_mat[idx_u][:, idx]  # (n*m, n_kept)
            V_r = V_mat[idx_v][:, idx]  # (m*p, n_kept)
            W_r = W_mat[idx_w][:, idx]  # (p*n, n_kept)
            
            U_reshaped = U_r.reshape(n, m, 1, -1)
            V_reshaped = V_r.reshape(1, m, p, -1)
            W_reshaped = W_r.reshape(n, 1, p, -1)
            
            result = np.sum(U_reshaped * V_reshaped * W_reshaped, axis=3)
            return np.allclose(result, 1.0, atol=1e-9)
        
        # Multiple passes of greedy removal
        for _pass in range(5):
            if time.time() - start_time > max_time:
                break
            removed_any = False
            for r in range(rank):
                if time.time() - start_time > max_time:
                    break
                if not keep[r]:
                    continue
                keep[r] = False
                if check_valid(keep):
                    removed_any = True
                else:
                    keep[r] = True
            if not removed_any:
                break
        
        idx = np.where(keep)[0]
        if len(idx) < rank:
            return U_mat[:, idx], V_mat[:, idx], W_mat[:, idx]
        return U_mat, V_mat, W_mat
    
    # Attempt rank reduction with generous time budget
    remaining_time = TIME_LIMIT - (time.time() - START_TIME) - 5.0
    U_opt, V_opt, W_opt = try_reduce_rank(U_np, V_np, W_np, max_time=max(5.0, remaining_time))
    
    # Convert back to lists with integer/half-integer entries if possible
    def to_list(mat):
        mat_rounded = np.round(mat * 2) / 2
        if np.allclose(mat, mat_rounded, atol=1e-9):
            return mat_rounded.tolist()
        else:
            return mat.tolist()
    
    return {"U": to_list(U_opt), "V": to_list(V_opt), "W": to_list(W_opt)}
# EVOLVE-BLOCK-END
