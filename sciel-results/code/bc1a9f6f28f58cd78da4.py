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
        rank_tensor = np.sum(s > tol)
        
        if rank_tensor < 66:
            # Find null space vectors
            # Null space is spanned by rows of Vt from index rank_tensor onward
            num_null = 66 - rank_tensor
            
            # Try to eliminate one column at a time
            # For each null vector v, we have sum_l v[l] * T_l = 0
            # So T_j = -sum_{l != j} (v[l]/v[j]) * T_l for any j with v[j] != 0
            # This means we can replace column j with a linear combination of others
            # and drop column j
            
            # Strategy: find a null vector, pick the column with largest |v[j]|,
            # express it as combination of others, fold into remaining columns
            
            for null_idx in range(num_null):
                v = Vt[rank_tensor + null_idx, :]
                
                # Find column to eliminate (largest absolute coefficient)
                abs_v = np.abs(v)
                max_idx = int(np.argmax(abs_v))
                
                if abs_v[max_idx] < 1e-12:
                    continue
                
                # T_max_idx = -sum_{l != max_idx} (v[l]/v[max_idx]) * T_l
                # So: sum_{l != max_idx} T_l - sum_{l != max_idx} (v[l]/v[max_idx]) * T_l = 0
                # New column l (for l != max_idx): T_l * (1 - v[l]/v[max_idx])
                # Wait, that's not right. Let me think again.
                # sum_l v[l] * T_l = 0
                # v[max_idx] * T_max_idx = -sum_{l != max_idx} v[l] * T_l
                # T_max_idx = -sum_{l != max_idx} (v[l]/v[max_idx]) * T_l
                # Original sum: sum_l T_l = sum_{l != max_idx} T_l + T_max_idx
                # = sum_{l != max_idx} T_l - sum_{l != max_idx} (v[l]/v[max_idx]) * T_l
                # = sum_{l != max_idx} (1 - v[l]/v[max_idx]) * T_l
                # So new coefficient for column l is (1 - v[l]/v[max_idx])
                
                # But wait - we can't just scale columns because the tensor is
                # a product U*V*W. Scaling a column means scaling U, V, or W.
                # If we scale column l by alpha_l, we can put sqrt(alpha_l) in U,
                # sqrt(alpha_l) in V, and 1/sqrt(alpha_l) in W, but only if
                # alpha_l > 0. For negative alpha_l, we need to be more careful.
                #
                # Actually, the simplest approach: just verify if the reduced
                # decomposition (dropping column max_idx and adjusting) works.
                # But adjusting requires scaling individual columns, which means
                # we need to modify U, V, W entries.
                #
                # Alternative simpler check: just drop the column and verify.
                # This works if T_max_idx = 0, which it's not in general.
                #
                # Correct approach: for each l != max_idx, the new "effective"
                # tensor is (1 - v[l]/v[max_idx]) * T_l. Since T_l = U_l * V_l * W_l
                # (outer product), scaling by a scalar means we can distribute the
                # scalar among U_l, V_l, W_l. E.g., multiply U_l by c, V_l by d,
                # W_l by e where c*d*e = (1 - v[l]/v[max_idx]).
                # Simplest: put all the scalar in one matrix.
                
                new_rank = 65
                U_new = np.zeros((n * m, new_rank))
                V_new = np.zeros((m * p, new_rank))
                W_new = np.zeros((p * n, new_rank))
                
                new_col = 0
                for l in range(66):
                    if l == max_idx:
                        continue
                    scale = 1.0 - v[l] / v[max_idx]
                    # Distribute scale: put it all in W (or U or V)
                    U_new[:, new_col] = U_np[:, l]
                    V_new[:, new_col] = V_np[:, l]
                    W_new[:, new_col] = W_np[:, l] * scale
                    new_col += 1
                
                # Verify the decomposition
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
                    # Success! Convert to lists
                    U_out = U_new.tolist()
                    V_out = V_new.tolist()
                    W_out = W_new.tolist()
                    return {"U": U_out, "V": V_out, "W": W_out}
    except Exception:
        pass
    
    # Return the rank-66 decomposition
    return {"U": U66, "V": V66, "W": W66}
