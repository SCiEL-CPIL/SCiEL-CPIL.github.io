import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 = 28
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen local matrices for 2x2x2
    # U_L[i, k] where i is local A index (0..3), k is local product index (0..6)
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_L = [
        [1, 0, 1, 0, 1, -1, 0], # a11
        [0, 0, 0, 0, 1, 0, 1],  # a12
        [0, 1, 0, 0, 0, 1, 0],  # a21
        [1, 1, 0, 1, 0, 0, -1]  # a22
    ]
    
    # V_L[j, k] where j is local B index (0..3), k is local product index (0..6)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V_L = [
        [1, 1, 0, -1, 0, 1, 0], # b11
        [0, 0, 1, 0, 0, 1, 0],  # b12
        [0, 0, 0, 1, 0, 0, 1],  # b21
        [1, 0, -1, 0, 1, 0, 1]  # b22
    ]
    
    # W_L[c, k] where c is local C index (0..3), k is local product index (0..6)
    # C indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) 
    # Note: W index is k*n + i. 
    # Local n=2.
    # c=0 -> k=0, i=0
    # c=1 -> k=1, i=0
    # c=2 -> k=0, i=1
    # c=3 -> k=1, i=1
    W_L = [
        [1, 0, 0, 1, -1, 0, 1],  # c11
        [0, 0, 1, 0, 1, 0, 0],   # c12
        [0, 1, 0, 1, 0, 0, 0],   # c21
        [1, -1, 1, 0, 0, 1, 0]   # c22
    ]
    
    # Define the 4 sub-blocks
    # Each sub-block is defined by:
    # u_rows: list of 4 global U row indices corresponding to local U rows 0,1,2,3
    # v_rows: list of 4 global V row indices corresponding to local V rows 0,1,2,3
    # w_rows: list of 4 global W row indices corresponding to local W rows 0,1,2,3
    # col_offset: starting column in U, V, W
    
    sub_blocks = []
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A rows: i=0,1; j=0,1
    # U global idx: i*4 + j
    u_rows_1 = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # 0, 1, 4, 5
    # B rows: j=0,1; k=0,1
    # V global idx: j*4 + k
    v_rows_1 = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # 0, 1, 4, 5
    # C cols: k=0,1; i=0,1
    # W global idx: k*2 + i
    w_rows_1 = [0*2+0, 1*2+0, 0*2+1, 1*2+1] # 0, 2, 1, 3 ? 
    # Wait, W_L order is c11(0,0), c12(1,0), c21(0,1), c22(1,1)
    # My W_L definition:
    # Row 0: c11 (k=0, i=0) -> idx 0
    # Row 1: c12 (k=1, i=0) -> idx 1
    # Row 2: c21 (k=0, i=1) -> idx 2
    # Row 3: c22 (k=1, i=1) -> idx 3
    # So w_rows_1 should map local row 0->global 0, 1->global 1, 2->global 2, 3->global 3.
    w_rows_1 = [0, 1, 2, 3]
    sub_blocks.append((u_rows_1, v_rows_1, w_rows_1, 0))
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A rows: i=0,1; j=2,3
    u_rows_2 = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # 2, 3, 6, 7
    # B rows: j=2,3; k=0,1
    v_rows_2 = [2*4+0, 2*4+1, 3*4+0, 3*4+1] # 8, 9, 12, 13
    # C cols: k=0,1; i=0,1 (Same as Block 1)
    w_rows_2 = [0, 1, 2, 3]
    sub_blocks.append((u_rows_2, v_rows_2, w_rows_2, 7))
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A rows: i=0,1; j=0,1 (Same as Block 1)
    u_rows_3 = [0, 1, 4, 5]
    # B rows: j=0,1; k=2,3
    v_rows_3 = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # 2, 3, 6, 7
    # C cols: k=2,3; i=0,1
    # W global idx: k*2 + i
    # c11 (local 0,0) -> global k=2, i=0 -> 2*2+0 = 4
    # c12 (local 1,0) -> global k=3, i=0 -> 3*2+0 = 6
    # c21 (local 0,1) -> global k=2, i=1 -> 2*2+1 = 5
    # c22 (local 1,1) -> global k=3, i=1 -> 3*2+1 = 7
    w_rows_3 = [4, 6, 5, 7]
    sub_blocks.append((u_rows_3, v_rows_3, w_rows_3, 14))
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A rows: i=0,1; j=2,3 (Same as Block 2)
    u_rows_4 = [2, 3, 6, 7]
    # B rows: j=2,3; k=2,3
    v_rows_4 = [2*4+2, 2*4+3, 3*4+2, 3*4+3] # 10, 11, 14, 15
    # C cols: k=2,3; i=0,1 (Same as Block 3)
    w_rows_4 = [4, 6, 5, 7]
    sub_blocks.append((u_rows_4, v_rows_4, w_rows_4, 21))
    
    # Fill the matrices
    for u_rows, v_rows, w_rows, col_offset in sub_blocks:
        for k in range(7):
            col = col_offset + k
            for local_idx in range(4):
                U[u_rows[local_idx]][col] += U_L[local_idx][k]
                V[v_rows[local_idx]][col] += V_L[local_idx][k]
                W[w_rows[local_idx]][col] += W_L[local_idx][k]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
