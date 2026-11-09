import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition fallback (rank = 90)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    # Split dimensions into blocks to apply Strassen (2x2x2) where possible
    # A: 3x5 -> rows: [2, 1], cols: [2, 2, 1]
    # B: 5x6 -> rows: [2, 2, 1], cols: [2, 2, 2]
    a_row_splits = [2, 1]
    a_col_splits = [2, 2, 1]
    b_col_splits = [2, 2, 2]
    
    if sum(a_row_splits) != n or sum(a_col_splits) != m or sum(b_col_splits) != p:
        return {"U": U_school, "V": V_school, "W": W_school}
        
    # Helper to get global indices
    def get_a_indices(row_start, col_start, r_size, c_size):
        indices = []
        for i in range(r_size):
            for j in range(c_size):
                indices.append((row_start + i) * m + (col_start + j))
        return indices

    def get_b_indices(row_start, col_start, r_size, c_size):
        indices = []
        for j in range(r_size):
            for k in range(c_size):
                indices.append((row_start + j) * p + (col_start + k))
        return indices

    def get_c_indices(row_start, col_start, r_size, c_size):
        # W index = k*n + i
        indices = []
        for k in range(c_size):
            for i in range(r_size):
                indices.append((col_start + k) * n + (row_start + i))
        return indices

    # Strassen 2x2x2 decomposition matrices
    # U rows (m1..m7), cols (a,b,c,d)
    U_strassen = [
        [1, 0, 0, 1],  # m1: a+d
        [0, 0, 1, 1],  # m2: c+d
        [1, 0, 0, 0],  # m3: a
        [0, 0, 0, 1],  # m4: d
        [1, 1, 0, 0],  # m5: a+b
        [0, 0, 1, -1], # m6: c-a
        [0, 1, 0, -1]  # m7: b-d
    ]
    
    # V rows (m1..m7), cols (e,f,g,h)
    V_strassen = [
        [1, 0, 0, 1],  # m1: e+h
        [1, 0, 0, 0],  # m2: e
        [0, 1, 0, -1], # m3: f-h
        [0, 0, 1, -1], # m4: g-e
        [0, 0, 0, 1],  # m5: h
        [1, 1, 0, 0],  # m6: e+f
        [0, 0, 1, 1]   # m7: g+h
    ]
    
    # W rows (C_00, C_10, C_01, C_11), cols (m1..m7)
    # C_00 = m1 + m4 - m5 + m7
    # C_10 = m2 + m4
    # C_01 = m3 + m5
    # C_11 = m1 - m2 + m3 + m6
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook for small blocks
    def get_schoolbook_block(r_size, c_size, k_size):
        rank = r_size * c_size * k_size
        U = [[0]*rank for _ in range(r_size*c_size)]
        V = [[0]*rank for _ in range(c_size*k_size)]
        W = [[0]*rank for _ in range(k_size*r_size)]
        
        col = 0
        for i in range(r_size):
            for j in range(c_size):
                for k in range(k_size):
                    U[i*c_size + j][col] = 1
                    V[j*k_size + k][col] = 1
                    W[k*r_size + i][col] = 1
                    col += 1
        return U, V, W

    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Iterate through blocks
    a_row_start = 0
    for i_block in range(len(a_row_splits)):
        r_size = a_row_splits[i_block]
        a_col_start = 0
        b_row_start = 0
        for j_block in range(len(a_col_splits)):
            c_size = a_col_splits[j_block]
            b_col_start = 0
            for k_block in range(len(b_col_splits)):
                k_size = b_col_splits[k_block]
                
                a_indices = get_a_indices(a_row_start, a_col_start, r_size, c_size)
                b_indices = get_b_indices(b_row_start, b_col_start, c_size, k_size)
                c_indices = get_c_indices(a_row_start, b_col_start, r_size, k_size)
                
                if r_size == 2 and c_size == 2 and k_size == 2:
                    U_s, V_s, W_s = U_strassen, V_strassen, W_strassen
                else:
                    U_s, V_s, W_s = get_schoolbook_block(r_size, c_size, k_size)
                
                num_cols = len(U_s[0])
                
                for col_idx in range(num_cols):
                    u_col = [0] * (n*m)
                    v_col = [0] * (m*p)
                    w_col = [0] * (p*n)
                    
                    for local_idx, global_idx in enumerate(a_indices):
                        u_col[global_idx] = U_s[local_idx][col_idx]
                    
                    for local_idx, global_idx in enumerate(b_indices):
                        v_col[global_idx] = V_s[local_idx][col_idx]
                    
                    for local_idx, global_idx in enumerate(c_indices):
                        w_col[global_idx] = W_s[local_idx][col_idx]
                        
                    U_blocks.append(u_col)
                    V_blocks.append(v_col)
                    W_blocks.append(w_col)
                
                b_col_start += k_size
            
            b_row_start += c_size
            a_col_start += c_size
        
        a_row_start += r_size

    if U_blocks:
        U = list(zip(*U_blocks)) # Transpose to row-major (n*m, rank)
        V = list(zip(*V_blocks))
        W = list(zip(*W_blocks))
        # Convert tuples to lists
        U = [list(row) for row in U]
        V = [list(row) for row in V]
        W = [list(row) for row in W]
        
        # Verify correctness
        rank = len(U[0])
        
        # Build tensor check
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # T[i*m+j, j*p+k, k*n+i] should be sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We want this to be 1 if the indices are consistent with matrix multiplication, else 0.
        # Actually, the tensor is T_{i,j,k} = 1.
        # Let's check a random sample of entries to be fast.
        
        valid = True
        # Check all entries? 3*5*6 = 90 entries. Sum over rank (33). 2970 mults. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    val = np.dot(U_np[idx_u], V_np[idx_v])
                    # Wait, the decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                    # This is not a simple dot product of two vectors because there are 3 factors.
                    # It's a trilinear form.
                    # val = sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    
                    vec_u = U_np[idx_u]
                    vec_v = V_np[idx_v]
                    vec_w = W_np[idx_w]
                    
                    # Compute sum of products
                    prod = vec_u * vec_v
                    val = np.dot(prod, vec_w)
                    
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U, "V": V, "W": W}
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
