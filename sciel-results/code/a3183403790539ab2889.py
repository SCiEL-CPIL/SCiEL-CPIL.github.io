import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    n, m, p = 3, 4, 6
    
    # Rank-66 block decomposition:
    # A (3x4): rows 1+2, cols 2+2
    # B (4x6): rows 2+2, cols 2+2+2
    # C[0,0:2] = A[0,0:2]*B[0:2,0:2] + A[0,2:4]*B[2:4,0:2]  (1x2*2x2 + 1x2*2x2 = 4+4)
    # C[0,2:4] = A[0,0:2]*B[0:2,2:4] + A[0,2:4]*B[2:4,2:4]  (4+4)
    # C[0,4:6] = A[0,0:2]*B[0:2,4:6] + A[0,2:4]*B[2:4,4:6]  (4+4)
    # C[1:3,0:2] = A[1:3,0:2]*B[0:2,0:2] + A[1:3,2:4]*B[2:4,0:2]  (7+7)
    # C[1:3,2:4] = A[1:3,0:2]*B[0:2,2:4] + A[1:3,2:4]*B[2:4,2:4]  (7+7)
    # C[1:3,4:6] = A[1:3,0:2]*B[0:2,4:6] + A[1:3,2:4]*B[2:4,4:6]  (7+7)
    # Total: 6*4 + 6*7 = 24 + 42 = 66
    
    U66 = [[0] * 66 for _ in range(n * m)]
    V66 = [[0] * 66 for _ in range(m * p)]
    W66 = [[0] * 66 for _ in range(p * n)]
    
    col = 0
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1
        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1
        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1
        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
        col += 1
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1
    
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)
    
    # Convert to numpy for linear algebra
    U_np = np.array(U66, dtype=float)
    V_np = np.array(V66, dtype=float)
    W_np = np.array(W66, dtype=float)
    
    # Build the tensor matrix: each column l is a flattened rank-1 tensor
    # T_l[i,j,k] = U[i,j,l] * V[j,k,l] * W[k,i,l]
    # We need sum_l T_l[i,j,k] = delta_{j,k}
    # If the 66 columns are linearly dependent, we can reduce rank
    
    tensor_matrix = np.zeros((n * m * p, 66))
    for l in range(66):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_matrix[i * m * p + j * p + k, l] = \
                        U_np[i * m + j, l] * V_np[j * p + k, l] * W_np[k * n + i, l]
    
    # Check for linear dependence using SVD
    try:
        U_svd, s, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
        # Count significant singular values
        tol = s[0] * max(tensor_matrix.shape) * np.finfo(float).eps
        rank_tensor = int(np.sum(s > tol))
        
        if rank_tensor < 66:
            num_null = 66 - rank_tensor
            null_space = Vt[rank_tensor:, :]  # shape (num_null, 66)
            
            # For each null vector v, sum_l v[l] * T_l = 0
            # We can eliminate column 'elim' by rescaling others:
            # T_elim = -sum_{l != elim} (v[l]/v[elim]) * T_l
            # New coefficient for T_l (l != elim): 1 - v[l]/v[elim]
            # This works because: sum_{l != elim} (1 - v[l]/v[elim]) T_l 
            #   = sum_{l != elim} T_l - (1/v[elim]) * sum_{l != elim} v[l] T_l
            #   = sum_{l != elim} T_l - (1/v[elim]) * (-v[elim] T_elim)
            #   = sum_{l != elim} T_l + T_elim
            #   = sum_{all l} T_l  (original)
            
            best_result = None
            best_err = 1e9
            
            for null_idx in range(num_null):
                if time.time() > deadline - 10:
                    break
                v = null_space[null_idx, :]
                abs_v = np.abs(v)
                
                # Try eliminating each column with significant coefficient
                candidates = np.where(abs_v > 1e-10)[0]
                
                for max_idx in candidates:
                    if time.time() > deadline - 5:
                        break
                    
                    new_rank = 65
                    U_new = np.zeros((n * m, new_rank))
                    V_new = np.zeros((m * p, new_rank))
                    W_new = np.zeros((p * n, new_rank))
                    
                    new_col = 0
                    valid = True
                    for l in range(66):
                        if l == max_idx:
                            continue
                        scale = 1.0 - v[l] / v[max_idx]
                        if abs(scale) < 1e-15:
                            valid = False
                            break
                        U_new[:, new_col] = U_np[:, l]
                        V_new[:, new_col] = V_np[:, l]
                        W_new[:, new_col] = W_np[:, l] * scale
                        new_col += 1
                    
                    if not valid or new_col != new_rank:
                        continue
                    
                    # Verify
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s_val = 0.0
                                for l in range(new_rank):
                                    s_val += U_new[i * m + j, l] * V_new[j * p + k, l] * W_new[k * n + i, l]
                                expected = 1.0 if j == k else 0.0
                                err = max(err, abs(s_val - expected))
                    
                    if err < best_err:
                        best_err = err
                        if err < 1e-9:
                            best_result = (U_new, V_new, W_new)
                            break
                if best_result is not None:
                    break
            
            # Try iterative elimination: after eliminating one column, check if more
            # columns can be eliminated from the reduced set
            if best_result is None:
                # Try greedy iterative elimination
                U_cur = U_np.copy()
                V_cur = V_np.copy()
                W_cur = W_np.copy()
                active = list(range(66))
                
                while len(active) > 0 and time.time() < deadline - 15:
                    # Rebuild tensor matrix for active columns
                    tensor_active = np.zeros((n * m * p, len(active)))
                    for idx, l in enumerate(active):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    tensor_active[i * m * p + j * p + k, idx] = \
                                        U_cur[i * m + j, l] * V_cur[j * p + k, l] * W_cur[k * n + i, l]
                    
                    # SVD to find null space
                    try:
                        Ua, sa, Vta = np.linalg.svd(tensor_active, full_matrices=False)
                        tol_a = sa[0] * max(tensor_active.shape) * np.finfo(float).eps
                        rank_a = int(np.sum(sa > tol_a))
                    except:
                        break
                    
                    if rank_a >= len(active):
                        break  # No more null vectors
                    
                    nulls_a = Vta[rank_a:, :]
                    eliminated = False
                    
                    for nidx in range(nulls_a.shape[0]):
                        if time.time() > deadline - 10:
                            break
                        v = nulls_a[nidx, :]
                        abs_v = np.abs(v)
                        candidates_a = np.where(abs_v > 1e-10)[0]
                        
                        for elim_idx in candidates_a:
                            if time.time() > deadline - 5:
                                break
                            elim_col = active[elim_idx]
                            
                            # Build new active set
                            new_active = [l for idx, l in enumerate(active) if idx != elim_idx]
                            new_rank = len(new_active)
                            
                            U_new = np.zeros((n * m, new_rank))
                            V_new = np.zeros((m * p, new_rank))
                            W_new = np.zeros((p * n, new_rank))
                            
                            new_col = 0
                            valid = True
                            for idx, l in enumerate(active):
                                if idx == elim_idx:
                                    continue
                                scale = 1.0 - v[idx] / v[elim_idx]
                                if abs(scale) < 1e-15:
                                    valid = False
                                    break
                                U_new[:, new_col] = U_cur[:, l]
                                V_new[:, new_col] = V_cur[:, l]
                                W_new[:, new_col] = W_cur[:, l] * scale
                                new_col += 1
                            
                            if not valid or new_col != new_rank:
                                continue
                            
                            # Verify
                            err = 0.0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        s_val = 0.0
                                        for l in range(new_rank):
                                            s_val += U_new[i * m + j, l] * V_new[j * p + k, l] * W_new[k * n + i, l]
                                        expected = 1.0 if j == k else 0.0
                                        err = max(err, abs(s_val - expected))
                            
                            if err < 1e-9:
                                U_cur = U_new
                                V_cur = V_new
                                W_cur = W_new
                                active = new_active
                                eliminated = True
                                best_result = (U_new, V_new, W_new)
                                break
                        if eliminated:
                            break
                    if not eliminated:
                        break
                
                if best_result is not None:
                    U_out = best_result[0].tolist()
                    V_out = best_result[1].tolist()
                    W_out = best_result[2].tolist()
                    return {"U": U_out, "V": V_out, "W": W_out}
            
            if best_result is not None:
                U_out = best_result[0].tolist()
                V_out = best_result[1].tolist()
                W_out = best_result[2].tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
    except Exception:
        pass
    
    # Return the rank-66 decomposition
    return {"U": U66, "V": V66, "W": W66}
