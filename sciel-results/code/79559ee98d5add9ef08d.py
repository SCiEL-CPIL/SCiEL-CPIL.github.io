import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Construct 4x4 factors using Kronecker product
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # The Kronecker product rows are indexed by (i1, j1, i2, j2)
    # Standard indices are i, j.
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # We need to map standard indices (i, j) to Kronecker indices.
    # kron_index = ((i1 * 2 + j1) * 2 + i2) * 2 + j2 ? No.
    # np.kron(A, B) has shape (4, 49).
    # Row r in kron corresponds to (r_outer, r_inner) where r = r_outer * 4 + r_inner.
    # r_outer is row in A (0-3), r_inner is row in B (0-3).
    # A row r_outer corresponds to (i1, j1) where r_outer = i1 * 2 + j1.
    # B row r_inner corresponds to (i2, j2) where r_inner = i2 * 2 + j2.
    # So kron row L = (i1 * 2 + j1) * 4 + (i2 * 2 + j2).
    
    # We want to fill U4[standard_idx, :] = U4_kron[kron_idx, :]
    # standard_idx = i * 4 + j.
    # i = 2 * i1 + i2
    # j = 2 * j1 + j2
    
    # We can just iterate over all standard indices and find the corresponding kron index.
    perm_U = np.zeros(16, dtype=int)
    perm_V = np.zeros(16, dtype=int)
    perm_W = np.zeros(16, dtype=int)
    
    for i in range(4):
        for j in range(4):
            std_idx = i * 4 + j
            # Solve for i1, j1, i2, j2
            i1 = i // 2
            i2 = i % 2
            j1 = j // 2
            j2 = j % 2
            
            kron_idx = (i1 * 2 + j1) * 4 + (i2 * 2 + j2)
            perm_U[std_idx] = kron_idx
            perm_V[std_idx] = kron_idx
            
            # W index is k * 4 + i.
            # Here the "standard" index for W in the tensor identity is k, i.
            # In the Kronecker product, W corresponds to (k1, i1) and (k2, i2).
            # k = 2 * k1 + k2
            # i = 2 * i1 + i2
            # The row in W4_kron is indexed by (k1 * 2 + i1) * 4 + (k2 * 2 + i2).
            # But wait, W2 factors are for (k, i) or (i, k)?
            # In Strassen 2x2:
            # C_ik = sum U_ij V_jk W_ki?
            # Usually W is indexed by (k, i) to match C_ik.
            # So W2 row (k, i) corresponds to k1 * 2 + i1, etc.
            
            # For a given standard W index (k, i):
            # k1 = k // 2, k2 = k % 2
            # i1 = i // 2, i2 = i % 2
            # kron_idx_W = (k1 * 2 + i1) * 4 + (k2 * 2 + i2)
            
    # We need to fill perm_W for all 16 standard indices (k, i).
    for k in range(4):
        for i in range(4):
            std_idx_W = k * 4 + i
            k1 = k // 2
            k2 = k % 2
            i1 = i // 2
            i2 = i % 2
            
            kron_idx_W = (k1 * 2 + i1) * 4 + (k2 * 2 + i2)
            perm_W[std_idx_W] = kron_idx_W
            
    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]
    
    # Verification
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_row = U4[i * 4 + j]
                v_row = V4[j * 4 + k]
                w_row = W4[k * 4 + i]
                prod = np.sum(u_row * v_row * w_row)
                if prod != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        if time.time() - start_time > deadline:
             # Fallback
             n, m, p = 4, 4, 4
             rank = n * m * p
             U_f = [[0] * rank for _ in range(n * m)]
             V_f = [[0] * rank for _ in range(m * p)]
             W_f = [[0] * rank for _ in range(p * n)]
             col = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U_f[i*m+j][col] = 1
                         V_f[j*p+k][col] = 1
                         W_f[k*n+i][col] = 1
                         col += 1
             return {"U": U_f, "V": V_f, "W": W_f}
             
        return {
            "U": U4.tolist(),
            "V": V4.tolist(),
            "W": W4.tolist()
        }
    else:
        # Fallback Rank 64
        n, m, p = 4, 4, 4
        rank = n * m * p
        U_f = [[0] * rank for _ in range(n * m)]
        V_f = [[0] * rank for _ in range(m * p)]
        W_f = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][col] = 1
                    V_f[j*p+k][col] = 1
                    W_f[k*n+i][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}
