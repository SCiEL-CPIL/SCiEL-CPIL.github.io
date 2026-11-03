import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # We decompose the multiplication into 4 independent 2x4 * 4x2 sub-problems
    # corresponding to the 4 pairs of columns in the output matrix C (size 2x8).
    # Each 2x4 * 4x2 multiplication can be done with rank 14 (sum of two Strassen 2x2 multiplications).
    # Total rank = 4 * 14 = 56.
    
    # Strassen components for 2x2 matrix multiplication
    # U_local: 4x7, V_local: 4x7, W_local: 4x7
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0 (M1)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1
    W_local[3][0] = 1
    
    # r=1 (M2)
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1
    W_local[3][1] = -1
    
    # r=2 (M3)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1
    W_local[3][2] = 1
    
    # r=3 (M4)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1
    W_local[2][3] = 1
    
    # r=4 (M5)
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1
    W_local[1][4] = 1
    
    # r=5 (M6)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # r=6 (M7)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # There are 4 column pairs in the output (p=8, so k=0..7).
    # For each pair l (0, 1, 2, 3), we compute C[:, 2l:2l+2] = A * B[:, 2l:2l+2].
    # This is a 2x4 * 4x2 multiplication.
    # We split A into A_left (cols 0,1) and A_right (cols 2,3).
    # We split B_part (rows 0..3, cols 2l, 2l+1) into B_top (rows 0,1) and B_bottom (rows 2,3).
    # C_part = A_left * B_top + A_right * B_bottom.
    # Each term is a 2x2 * 2x2 multiplication, using 7 rank-1 tensors.
    # Total 14 rank-1 tensors per column pair.
    
    for l in range(4):
        # Base rank index for this column pair
        base_r = l * 14
        
        # --- Term 1: A_left * B_top ---
        # A_left indices in U: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
        # B_top indices in V: B[0, 2l]=2l, B[0, 2l+1]=2l+1, B[1, 2l]=8+2l, B[1, 2l+1]=8+2l+1
        # C_part indices in W: C[0, 2l]=2l*2+0=4l, C[0, 2l+1]=4l+2, C[1, 2l]=4l+1, C[1, 2l+1]=4l+3
        
        a_map_1 = [0, 1, 4, 5]
        b_map_1 = [2*l, 2*l+1, 8+2*l, 8+2*l+1]
        c_map = [4*l, 4*l+2, 4*l+1, 4*l+3] # C[0,0], C[0,1], C[1,0], C[1,1] relative to block
        
        for s in range(7):
            r = base_r + s
            for la in range(4):
                val = U_local[la][s]
                if val != 0:
                    U[a_map_1[la], r] += val
            for lb in range(4):
                val = V_local[lb][s]
                if val != 0:
                    V[b_map_1[lb], r] += val
            for lc in range(4):
                val = W_local[lc][s]
                if val != 0:
                    W[c_map[lc], r] += val

        # --- Term 2: A_right * B_bottom ---
        # A_right indices in U: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
        # B_bottom indices in V: B[2, 2l]=16+2l, B[2, 2l+1]=16+2l+1, B[3, 2l]=24+2l, B[3, 2l+1]=24+2l+1
        # C_part indices in W: Same as above, since it adds to the same C block.
        
        a_map_2 = [2, 3, 6, 7]
        b_map_2 = [16+2*l, 16+2*l+1, 24+2*l, 24+2*l+1]
        # c_map is the same
        
        for s in range(7):
            r = base_r + 7 + s
            for la in range(4):
                val = U_local[la][s]
                if val != 0:
                    U[a_map_2[la], r] += val
            for lb in range(4):
                val = V_local[lb][s]
                if val != 0:
                    V[b_map_2[lb], r] += val
            for lc in range(4):
                val = W_local[lc][s]
                if val != 0:
                    W[c_map[lc], r] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
