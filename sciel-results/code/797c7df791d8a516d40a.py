import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 145.0 # 150s limit, keep 5s margin
    
    # 1. Generate 2x2x2 Strassen decomposition
    # U, V, W of shape (4, 7)
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_222[0,0]=1; U_222[3,0]=1
    V_222[0,0]=1; V_222[3,0]=1
    W_222[0,0]=1; W_222[1,0]=1; W_222[2,0]=1; W_222[3,0]=1
    
    # M2 = (A10+A11)B00
    U_222[2,1]=1; U_222[3,1]=1
    V_222[0,1]=1
    W_222[2,1]=1; W_222[3,1]=-1
    
    # M3 = A00(B01-B11)
    U_222[0,2]=1
    V_222[1,2]=1; V_222[3,2]=-1
    W_222[1,2]=1; W_222[3,2]=1
    
    # M4 = A11(B10-B00)
    U_222[3,3]=1
    V_222[2,3]=1; V_222[0,3]=-1
    W_222[0,3]=1; W_222[2,3]=1
    
    # M5 = (A00+A01)B11
    U_222[0,4]=1; U_222[1,4]=1
    V_222[3,4]=1
    W_222[0,4]=-1; W_222[1,4]=1
    
    # M6 = (A10-A00)(B00+B01)
    U_222[2,5]=1; U_222[0,5]=-1
    V_222[0,5]=1; V_222[1,5]=1
    W_222[3,5]=1
    
    # M7 = (A01-A11)(B10+B11)
    U_222[1,6]=1; U_222[3,6]=-1
    V_222[2,6]=1; V_222[3,6]=1
    W_222[0,6]=-1

    # 2. Generate 4x4x4 Strassen decomposition
    # U_444 shape (16, 49), V_444 shape (16, 49), W_444 shape (16, 49)
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Define Strassen terms for 4x4
    # Each term k has:
    # A_blocks: list of (block_index, coeff)
    # B_blocks: list of (block_index, coeff)
    # C_blocks: list of (block_index, coeff)
    
    # Block indices for 4x4 matrix (2x2 blocks of 2x2):
    # 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
    # Row offset for block b in 4x4 matrix (row-major 2x2 blocks):
    # b=0: rows 0-3 (in 4x4 flattened? No, U index is i*m+j.
    # For A (4x4), U index is i*4+j.
    # Block 0 (rows 0-1, cols 0-1): indices 0,1,4,5
    # Block 1 (rows 0-1, cols 2-3): indices 2,3,6,7
    # Block 2 (rows 2-3, cols 0-1): indices 8,9,12,13
    # Block 3 (rows 2-3, cols 2-3): indices 10,11,14,15
    
    # Let's define the mapping from block index to local 2x2 indices in U
    # U index = i*4+j.
    # Block 0: i in 0,1; j in 0,1.
    # Local 2x2 index l = i_loc*2+j_loc.
    # Global index = (i_loc+row_off)*4 + (j_loc+col_off).
    
    def get_global_indices(block_idx):
        row_off = (block_idx // 2) * 2
        col_off = (block_idx % 2) * 2
        indices = []
        for i_loc in range(2):
            for j_loc in range(2):
                i = row_off + i_loc
                j = col_off + j_loc
                indices.append(i * 4 + j)
        return indices

    # Strassen terms for 4x4
    # M1 = (A00+A11)(B00+B11) -> C00, C01, C10, C11 all get M1
    terms = [
        {
            "A": [(0, 1), (3, 1)],
            "B": [(0, 1), (3, 1)],
            "C": [(0, 1), (1, 1), (2, 1), (3, 1)]
        },
        {
            "A": [(2, 1), (3, 1)],
            "B": [(0, 1)],
            "C": [(2, 1), (3, -1)]
        },
        {
            "A": [(0, 1)],
            "B": [(1, 1), (3, -1)],
            "C": [(1, 1), (3, 1)]
        },
        {
            "A": [(3, 1)],
            "B": [(2, 1), (0, -1)],
            "C": [(0, 1), (2, 1)]
        },
        {
            "A": [(0, 1), (1, 1)],
            "B": [(3, 1)],
            "C": [(0, -1), (1, 1)]
        },
        {
            "A": [(2, 1), (0, -1)],
            "B": [(0, 1), (1, 1)],
            "C": [(3, 1)]
        },
        {
            "A": [(1, 1), (3, -1)],
            "B": [(2, 1), (3, 1)],
            "C": [(0, -1)]
        }
    ]

    for k in range(7):
        term = terms[k]
        for s in range(7):
            col = k * 7 + s
            # U part
            for b_idx, coeff in term["A"]:
                g_indices = get_global_indices(b_idx)
                for l in range(4):
                    val = U_222[l, s]
                    if val != 0:
                        U_444[g_indices[l], col] += coeff * val
            # V part
            for b_idx, coeff in term["B"]:
                g_indices = get_global_indices(b_idx)
                for l in range(4):
                    val = V_222[l, s]
                    if val != 0:
                        V_444[g_indices[l], col] += coeff * val
            # W part
            for b_idx, coeff in term["C"]:
                g_indices = get_global_indices(b_idx)
                for l in range(4):
                    val = W_222[l, s]
                    if val != 0:
                        W_444[g_indices[l], col] += coeff * val

    # 3. Generate 4x4x1 Schoolbook decomposition
    # U_441 shape (16, 16), V_441 shape (4, 16), W_441 shape (4, 16)
    U_441 = np.zeros((16, 16))
    V_441 = np.zeros((4, 16))
    W_441 = np.zeros((4, 16))
    
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U_441[i * 4 + j, c] = 1
            V_441[j, c] = 1
            W_441[i, c] = 1

    # 4. Assemble final matrices
    R = 49 + 49 + 16
    U = np.zeros((16, R))
    V = np.zeros((36, R))
    W = np.zeros((36, R))
    
    # Part 1: B1 (cols 0-3 of B, rows 0-15 of V, rows 0-15 of W)
    U[:, 0:49] = U_444
    V[0:16, 0:49] = V_444
    W[0:16, 0:49] = W_444
    
    # Part 2: B2 (cols 4-7 of B, rows 16-31 of V, rows 16-31 of W)
    U[:, 49:98] = U_444
    V[16:32, 49:98] = V_444
    W[16:32, 49:98] = W_444
    
    # Part 3: B3 (col 8 of B, row 32-35 of V, rows 32-35 of W)
    U[:, 98:114] = U_441
    V[32:36, 98:114] = V_441
    W[32:36, 98:114] = W_441
    
    return {"U": U, "V": V, "W": W}

