import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 4, 8
    total_rank = 56
    
    # Initialize matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen factors for 2x2x2
    # r, u_idx, u_val, v_idx, v_val, w_idx, w_val
    # Indices are local to the 2x2 block (0..3)
    strassen_factors = [
        # r=0
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # r=1
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # r=2
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # r=3
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # r=4
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # r=5
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # r=6
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]

    # We have 2 halves of A (cols 0-1, 2-3)
    # For each half, we have 4 blocks of B (cols 0-1, 2-3, 4-5, 6-7)
    # Total 8 blocks.
    
    block_id = 0
    for half in range(2):
        # A cols: 2*half, 2*half+1
        a_col_start = 2 * half
        
        for b_block in range(4):
            # B rows: 2*half, 2*half+1 (corresponding to A half)
            # B cols: 2*b_block, 2*b_block+1
            b_row_start = 2 * half
            b_col_start = 2 * b_block
            
            # C cols: 2*b_block, 2*b_block+1
            c_col_start = 2 * b_block
            
            base_r = block_id * 7
            
            for r in range(7):
                rr = base_r + r
                u_idx, u_val, v_idx, v_val, w_idx, w_val = strassen_factors[r]
                
                for idx, val in zip(u_idx, u_val):
                    # Local A index idx -> Global U index
                    # idx 0: A[0, 2*half] -> 0*m + 2*half
                    # idx 1: A[0, 2*half+1] -> 0*m + 2*half+1
                    # idx 2: A[1, 2*half] -> 1*m + 2*half
                    # idx 3: A[1, 2*half+1] -> 1*m + 2*half+1
                    local_i = idx // 2
                    local_j = idx % 2
                    global_u_idx = local_i * m + (a_col_start + local_j)
                    U[global_u_idx, rr] += val

                for idx, val in zip(v_idx, v_val):
                    # Local B index idx -> Global V index
                    # idx 0: B[2*half, 2*b_block] -> (2*half)*p + 2*b_block
                    # idx 1: B[2*half, 2*b_block+1] -> (2*half)*p + 2*b_block+1
                    # idx 2: B[2*half+1, 2*b_block] -> (2*half+1)*p + 2*b_block
                    # idx 3: B[2*half+1, 2*b_block+1] -> (2*half+1)*p + 2*b_block+1
                    local_i = idx // 2
                    local_j = idx % 2
                    global_v_idx = (b_row_start + local_i) * p + (b_col_start + local_j)
                    V[global_v_idx, rr] += val

                for idx, val in zip(w_idx, w_val):
                    # Local C index idx -> Global W index
                    # W index = k*n + i
                    # idx 0: C[0, 2*b_block] -> 0*n + 2*b_block
                    # idx 1: C[0, 2*b_block+1] -> 0*n + 2*b_block+1
                    # idx 2: C[1, 2*b_block] -> 1*n + 2*b_block
                    # idx 3: C[1, 2*b_block+1] -> 1*n + 2*b_block+1
                    local_i = idx // 2
                    local_j = idx % 2
                    global_w_idx = (c_col_start + local_j) * n + local_i
                    W[global_w_idx, rr] += val

            block_id += 1

    # Convert to lists if required by interface, though numpy arrays are often accepted
    # The seed returns lists of lists. I will return lists to be safe, or check if numpy is ok.
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # It doesn't specify type, but the seed uses lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
