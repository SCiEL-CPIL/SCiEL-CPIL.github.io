import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor decomposition.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition block by block.
    # Block structure:
    # j split: 2, 2, 1
    # k split: 2, 2, 2
    # i is always 2
    
    # Blocks:
    # (j_start, k_start, j_size, k_size)
    blocks = [
        (0, 0, 2, 2),
        (0, 2, 2, 2),
        (0, 4, 2, 2),
        (2, 0, 2, 2),
        (2, 2, 2, 2),
        (2, 4, 2, 2),
        (4, 0, 1, 2),
        (4, 2, 1, 2),
        (4, 4, 1, 2),
    ]
    
    # Initialize U, V, W as lists of lists (rows x cols)
    # We don't know the total rank yet, so we'll use lists of tuples (row_idx, value) for each column?
    # Or just build the columns and then assemble.
    
    columns_U = []
    columns_V = []
    columns_W = []
    
    def add_strassen_block(j_start, k_start, i_start=0):
        # 2x2x2 Strassen
        # A indices: (i_start, j_start) to (i_start+1, j_start+1)
        # B indices: (j_start, k_start) to (j_start+1, k_start+1)
        # C indices: (k_start, i_start) to (k_start+1, i_start+1) in W's index scheme (k, i)
        
        # Strassen terms:
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        # M2: (A10+A11)B00 -> C10 += 1, C11 -= 1
        # M3: A00(B01-B11) -> C01 += 1, C11 += 1
        # M4: A11(B10-B00) -> C00 += 1, C10 += 1
        # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
        # M6: (A10-A00)(B00+B01) -> C11 += 1
        # M7: (A01-A11)(B10+B11) -> C00 += 1
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u: { (i, j): coeff }
            # v: { (j, k): coeff }
            # w: { (k, i): coeff }
            
            # M1
            (
                {(0,0): 1, (1,1): 1},
                {(0,0): 1, (1,1): 1},
                {(0,0): 1, (1,1): 1}
            ),
            # M2
            (
                {(1,0): 1, (1,1): 1},
                {(0,0): 1},
                {(0,1): 1, (1,1): -1} # C10, C11
            ),
            # M3
            (
                {(0,0): 1},
                {(0,1): 1, (1,1): -1},
                {(1,0): 1, (1,1): 1} # C01, C11
            ),
            # M4
            (
                {(1,1): 1},
                {(1,0): 1, (0,0): -1},
                {(0,0): 1, (0,1): 1} # C00, C10
            ),
            # M5
            (
                {(0,0): 1, (0,1): 1},
                {(1,1): 1},
                {(0,0): -1, (1,0): 1} # C00, C01
            ),
            # M6
            (
                {(1,0): 1, (0,0): -1},
                {(0,0): 1, (0,1): 1},
                {(1,1): 1} # C11
            ),
            # M7
            (
                {(0,1): 1, (1,1): -1},
                {(1,0): 1, (1,1): 1},
                {(0,0): 1} # C00
            ),
        ]
        
        for u_dict, v_dict, w_dict in terms:
            u_col = [0.0] * (n * m)
            v_col = [0.0] * (m * p)
            w_col = [0.0] * (p * n)
            
            for (di, dj), val in u_dict.items():
                i_idx = i_start + di
                j_idx = j_start + dj
                u_col[i_idx * m + j_idx] = val
                
            for (dj, dk), val in v_dict.items():
                j_idx = j_start + dj
                k_idx = k_start + dk
                v_col[j_idx * p + k_idx] = val
                
            for (dk, di), val in w_dict.items():
                k_idx = k_start + dk
                i_idx = i_start + di
                w_col[k_idx * n + i_idx] = val
                
            columns_U.append(u_col)
            columns_V.append(v_col)
            columns_W.append(w_col)
            
    def add_outer_block(j_start, k_start, i_start=0):
        # 2x1x2 block
        # A: (i_start, j_start) to (i_start+1, j_start)
        # B: (j_start, k_start) to (j_start, k_start+1)
        # C: (k_start, i_start) to (k_start+1, i_start+1)
        
        # Term: A_ij B_jk -> C_ki
        # Rank 1:
        u_dict = {(0, 0): 1, (1, 0): 1}
        v_dict = {(0, 0): 1, (0, 1): 1}
        w_dict = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
        
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        
        for (di, dj), val in u_dict.items():
            i_idx = i_start + di
            j_idx = j_start + dj
            u_col[i_idx * m + j_idx] = val
            
        for (dj, dk), val in v_dict.items():
            j_idx = j_start + dj
            k_idx = k_start + dk
            v_col[j_idx * p + k_idx] = val
            
        for (dk, di), val in w_dict.items():
            k_idx = k_start + dk
            i_idx = i_start + di
            w_col[k_idx * n + i_idx] = val
            
        columns_U.append(u_col)
        columns_V.append(v_col)
        columns_W.append(w_col)

    for j_start, k_start, j_size, k_size in blocks:
        if time.time() > deadline:
            break
            
        if j_size == 2 and k_size == 2:
            add_strassen_block(j_start, k_start)
        elif j_size == 1 and k_size == 2:
            add_outer_block(j_start, k_start)
        else:
            # Fallback to schoolbook for unexpected sizes
            for di in range(j_size):
                for dk in range(k_size):
                    for di_c in range(n): # i
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        
                        i_idx = di_c
                        j_idx = j_start + di
                        k_idx = k_start + dk
                        
                        u_col[i_idx * m + j_idx] = 1.0
                        v_col[j_idx * p + k_idx] = 1.0
                        w_col[k_idx * n + i_idx] = 1.0
                        
                        columns_U.append(u_col)
                        columns_V.append(v_col)
                        columns_W.append(w_col)

    rank = len(columns_U)
    
    # Transpose to (rows, cols) format if needed?
    # The input lists are already rows (length n*m etc)
    # So U is list of rows.
    
    return {"U": columns_U, "V": columns_V, "W": columns_W}
# EVOLVE-BLOCK-END
