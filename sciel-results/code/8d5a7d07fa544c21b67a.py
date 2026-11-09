"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4 x 4x4 multiplication using Strassen (Rank 49)
    # Part 2: 4x4 x 4x2 multiplication using Schoolbook (Rank 32)
    # Total Rank = 81
    
    rank1 = 49
    rank2 = 32
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Strassen 2x2x2 Coefficients ---
    # U_strassen: 4x7
    # V_strassen: 4x7
    # W_strassen: 4x7
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    U_s = np.array([
        [1, 0, 0, 1, 1, 0, 0], # 00
        [0, 0, 1, 0, 0, 1, 0], # 01
        [0, 1, 0, 0, 0, 0, 1], # 10
        [1, 0, 0, 1, 1, 1, 1]  # 11
    ])
    
    V_s = np.array([
        [1, 1, 1, 0, 0, 0, 0], # 00
        [1, 0, 0, 1, 1, 1, 1], # 01
        [0, 1, 0, 1, 0, 1, 1], # 10
        [0, 0, 1, 0, 1, 0, 1]  # 11
    ])
    
    W_s = np.array([
        [1, -1, 0, 0, 1, 0, -1], # 00
        [0, 0, 1, 0, 0, 0, 1],   # 01
        [0, 1, 0, 1, 0, 0, 0],   # 10
        [1, -1, 1, 0, 0, 1, 0]   # 11
    ])
    
    # Outer Strassen (4x4x4)
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    U_outer = np.array([
        [1, 0, 0, 1], # 1
        [0, 0, 1, 1], # 2
        [1, 0, 0, 0], # 3
        [0, 0, 0, 1], # 4
        [1, 1, 0, 0], # 5
        [0, 0, 1, -1],# 6
        [0, 1, 0, -1] # 7
    ])
    
    V_outer = np.array([
        [1, 0, 0, 1], # 1
        [1, 0, 0, 0], # 2
        [0, 1, 0, -1],# 3
        [0, -1, 1, 0],# 4
        [0, 1, 0, 1], # 5
        [1, 1, 0, 0], # 6
        [0, 0, 1, 1]  # 7
    ])
    
    W_outer = np.array([
        [1, 0, 0, -1, 1, 0, -1], # C00
        [0, 0, 1, 0, 0, 0, 1],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ])
    
    # Fill U, V, W for Part 1 (Columns 0 to 48)
    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            
            # U: (i,j) from outer, (x,y) from inner
            # A index: (i*2+x, j*2+y) ? No.
            # A is 4x4. Block (i,j) is 2x2 at rows 2i..2i+1, cols 2j..2j+1.
            # Inner index (x,y) within block.
            # So global row = 2*i + x, col_in_block = 2*j + y.
            # But U is indexed by (row, col) of A?
            # A is 4x4. Flattened row-major: index = r*4 + c.
            # Block (i,j) covers rows 2i, 2i+1 and cols 2j, 2j+1.
            # Inner (x,y) corresponds to local row x, local col y.
            # Global row r = 2i + x.
            # Global col c = 2j + y.
            # But U has rows for A's elements.
            # Wait, the task says U has shape (n*m, rank).
            # A is n x m = 4x4.
            # So U row index corresponds to A's element (r, c) flattened.
            
            for i in range(2):
                for j in range(2):
                    # Outer coefficient
                    coeff_outer = U_outer[k, i*2+j] # i,j are block indices 0..3
                    if coeff_outer == 0: continue
                    
                    # Inner coefficients for this block
                    for x in range(2):
                        for y in range(2):
                            coeff_inner = U_s[l, x*2+y]
                            if coeff_inner == 0: continue
                            
                            val = coeff_outer * coeff_inner
                            # Global row in U
                            r = 2*i + x
                            c = 2*j + y
                            u_idx = r * 4 + c
                            U[u_idx][col] += val
    
            # V: (k,l) from outer, (z,w) from inner
            # B is 4x4 (first 4 cols).
            # Block (k,l) covers rows 2k..2k+1, cols 2l..2l+1.
            # Inner (z,w) local.
            # Global row r = 2k + z.
            # Global col c = 2l + w.
            # V row index = r * 4 + c.
            
            for k_idx in range(2):
                for l_idx in range(2):
                    coeff_outer = V_outer[k, k_idx*2+l_idx]
                    if coeff_outer == 0: continue
                    
                    for z in range(2):
                        for w in range(2):
                            coeff_inner = V_s[l, z*2+w]
                            if coeff_inner == 0: continue
                            
                            val = coeff_outer * coeff_inner
                            r = 2*k_idx + z
                            c = 2*l_idx + w
                            v_idx = r * 4 + c
                            V[v_idx][col] += val
    
            # W: (m,n) from outer, (u,v) from inner
            # C is 4x4 (first 4 cols of output).
            # Output block (m,n) covers rows 2m..2m+1, cols 2n..2n+1.
            # Inner (u,v) local.
            # W row index = (col_of_C * n + row_of_C) ?
            # Task: W[k*n+i, r].
            # C is 4x6. First part is 4x4.
            # Output element (r, c) where r,c in 0..3.
            # W index = c * 4 + r.
            
            for m_idx in range(2):
                for n_idx in range(2):
                    coeff_outer = W_outer[k, m_idx*2+n_idx]
                    if coeff_outer == 0: continue
                    
                    for u in range(2):
                        for v in range(2):
                            coeff_inner = W_s[l, u*2+v]
                            if coeff_inner == 0: continue
                            
                            val = coeff_outer * coeff_inner
                            r = 2*m_idx + u
                            c = 2*n_idx + v
                            w_idx = c * 4 + r
                            W[w_idx][col] += val

    # --- Part 2: 4x4 x 4x2 Schoolbook (Rank 32) ---
    # A is 4x4. B is 4x2 (cols 4,5).
    # C is 4x2 (cols 4,5).
    # Columns 49 to 80.
    
    for i in range(4):
        for j in range(4):
            for k in range(2):
                col = rank1 + (i * 4 * 2) + (j * 2) + k
                
                # U: A[i, j]
                u_idx = i * 4 + j
                U[u_idx][col] = 1
                
                # V: B[j, 4+k]
                # V rows correspond to B (4x6).
                # B row j, col 4+k.
                v_idx = j * 6 + (4 + k)
                V[v_idx][col] = 1
                
                # W: C[i, 4+k]
                # W rows correspond to C (4x6) flattened as c*4+r?
                # Task: W[k*n+i, r]. Here n=4.
                # Output col c = 4+k. Output row r = i.
                # w_idx = c * 4 + r.
                w_idx = (4 + k) * 4 + i
                W[w_idx][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
