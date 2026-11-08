import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    # Use the 4-block structure: A split into two 3x2 blocks, B split into four 2x2 blocks
    # Each block computes a 3x2 @ 2x2 = 3x2 multiplication
    # Rank 11 per block gives total rank 44
    # Try to see if we can reduce one block to rank 10 for total rank 43
    
    # For now, use the known rank 11 decomposition for 3x2 x 2x2
    # Local indices: A is 3x2 (indices 0-5), B is 2x2 (indices 0-3), C is 3x2 (indices 0-5)
    
    local_terms = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]),
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]),
        ([ (4, 1) ], [ (0, 1) ], [ (4, 1) ]),
        ([ (5, 1) ], [ (2, 1) ], [ (4, 1) ]),
        ([ (4, 1) ], [ (1, 1) ], [ (5, 1) ]),
        ([ (5, 1) ], [ (3, 1) ], [ (5, 1) ]),
    ]
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Blocks: (a_col_start, b_row_start, b_col_start, c_col_start)
    blocks = [
        (0, 0, 0, 0),  # A[:,0:2] @ B[0:2,0:2] -> C[:,0:2]
        (2, 2, 0, 0),  # A[:,2:4] @ B[2:4,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # A[:,0:2] @ B[0:2,2:4] -> C[:,2:4]
        (2, 2, 2, 2),  # A[:,2:4] @ B[2:4,2:4] -> C[:,2:4]
    ]
    
    for b_idx, (a_cs, b_rs, b_cs, c_cs) in enumerate(blocks):
        col_offset = b_idx * 11
        for l_idx, (u_list, v_list, w_list) in enumerate(local_terms):
            col = col_offset + l_idx
            
            for idx, val in u_list:
                r_loc = idx // 2
                c_loc = idx % 2
                g_a_idx = r_loc * m + (a_cs + c_loc)
                U[g_a_idx, col] += val
                
            for idx, val in v_list:
                r_loc = idx // 2
                c_loc = idx % 2
                g_b_idx = (b_rs + r_loc) * p + (b_cs + c_loc)
                V[g_b_idx, col] += val
                
            for idx, val in w_list:
                r_loc = idx // 2
                c_loc = idx % 2
                g_c_idx = (c_cs + c_loc) * n + r_loc
                W[g_c_idx, col] += val
                
        if time.time() > deadline:
            break
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
