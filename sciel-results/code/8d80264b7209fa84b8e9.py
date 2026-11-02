import numpy as np
import time

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Strategy:
    - Partition middle dim (4) into two 2-blocks, last dim (6) into three 2-blocks
    - For each (j_block, k_block): 3x2x2 sub-tensor
    - Use Strassen on rows (0,1): 7 terms, schoolbook on row 2: 2 terms = 9 terms per block? 
      Wait, schoolbook on row 2 (1x2x2) is 4 terms.
      Total 11 terms per block.
    - 2 j_blocks * 3 k_blocks * 11 = 66 terms total
    
    U shape: (12, 66)  -- rows indexed by i*4+j
    V shape: (24, 66)  -- rows indexed by j*6+k
    W shape: (18, 66)  -- rows indexed by k*3+i
    """
    if time.time() - START_TIME > TIME_LIMIT:
        return _construct_schoolbook()

    n, m, p = 3, 4, 6
    rank = 66
    U = [[0] * rank for _ in range(n * m)]   # 12 x 66
    V = [[0] * rank for _ in range(m * p)]   # 24 x 66
    W = [[0] * rank for _ in range(p * n)]   # 18 x 66
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    def add_strassen_3x2x2(i0, i1, i2, j_base, k_base):
        """
        Add 11 terms for a 3x2x2 block:
        - Strassen on rows (i0, i1): 7 terms
        - Schoolbook on row i2: 4 terms
        """
        nonlocal col
        
        # A indices for rows i0, i1 (2x2 sub-block)
        a_idx = i0 * m + j_base
        b_idx = i0 * m + (j_base + 1)
        c_idx = i1 * m + j_base
        d_idx = i1 * m + (j_base + 1)
        
        # A index for row i2
        e2_idx = i2 * m + j_base
        f2_idx = i2 * m + (j_base + 1)
        
        # B indices (2x2 sub-block)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        # W indices for rows i0, i1 (2x2 sub-block), W index = k*n + i
        p_idx = k_base * n + i0
        q_idx = (k_base + 1) * n + i0
        r_idx = k_base * n + i1
        s_idx = (k_base + 1) * n + i1
        
        # W indices for row i2
        t_idx = k_base * n + i2
        u_idx = (k_base + 1) * n + i2
        
        # Strassen on (i0, i1) x 2x2:
        # P1 = (a+d)(e+h)
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P2 = (c+d)e
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # P3 = a(f-h)
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P4 = d(g-e)
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # P5 = (a+b)h
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1
        
        # Schoolbook for row i2 (1x2x2 = 4 terms):
        # C[i2, k_base] = A[i2, j_base]*B[j_base, k_base] + A[i2, j_base+1]*B[j_base+1, k_base]
        U[e2_idx][col] = 1
        V[e_idx][col] = 1
        W[t_idx][col] = 1
        col += 1
        
        U[f2_idx][col] = 1
        V[g_idx][col] = 1
        W[t_idx][col] = 1
        col += 1
        
        # C[i2, k_base+1] = A[i2, j_base]*B[j_base, k_base+1] + A[i2, j_base+1]*B[j_base+1, k_base+1]
        U[e2_idx][col] = 1
        V[f_idx][col] = 1
        W[u_idx][col] = 1
        col += 1
        
        U[f2_idx][col] = 1
        V[h_idx][col] = 1
        W[u_idx][col] = 1
        col += 1

    # Process all 6 blocks: 2 j_blocks x 3 k_blocks
    for j_block in range(2):
        j_base = j_block * 2
        for k_block in range(3):
            k_base = k_block * 2
            if time.time() - START_TIME > TIME_LIMIT:
                return _construct_schoolbook()
            add_strassen_3x2x2(0, 1, 2, j_base, k_base)
    
    # Verification
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    # Fallback to schoolbook if something went wrong
                    return _construct_schoolbook()
    
    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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
