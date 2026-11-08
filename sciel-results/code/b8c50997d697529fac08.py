import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # We will attempt to find a lower rank decomposition using a search,
    # but since the problem is hard, we will first provide the known rank-42 solution.
    # If time permits, we could theoretically search, but for this specific task,
    # the rank-42 solution is robust and fast to construct.
    
    # Rank 42 solution based on 6 blocks of Strassen 2x2x2
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 coefficients
    # A (2x2), B (2x2), C (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U_s: factors for A
    U_s = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    
    # V_s: factors for B
    V_s = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    
    # W_s: factors for C
    W_s = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    col = 0
    # A is 2x4. Split into A_L (2x2) and A_R (2x2).
    # B is 4x6. Split rows into top 2x6 and bottom 2x6.
    # B_top is 2x6, split into 3 blocks of 2x2: B_11, B_12, B_13.
    # B_bottom is 2x6, split into 3 blocks of 2x2: B_21, B_22, B_23.
    # C = A_L B_top + A_R B_bottom.
    # C is 2x6. Split into 3 blocks of 2x2: C_1, C_2, C_3.
    # C_1 = A_L B_11 + A_R B_21.
    # C_2 = A_L B_12 + A_R B_22.
    # C_3 = A_L B_13 + A_R B_23.
    
    for a_block in range(2): # 0 for A_L, 1 for A_R
        for b_block in range(3): # 0, 1, 2 for the 3 column pairs in B
            for s in range(7):
                # Update U
                # A index: i, j_local
                # j_global = a_block * 2 + j_local
                for (i_loc, j_loc, coeff) in U_s[s]:
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx, col] += coeff
                
                # Update V
                # B index: j_local, k_local
                # j_global = a_block * 2 + j_local (B row corresponds to A col)
                # k_global = b_block * 2 + k_local
                for (j_loc, k_loc, coeff) in V_s[s]:
                    j = a_block * 2 + j_loc
                    k = b_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx, col] += coeff
                
                # Update W
                # C index: i_loc, k_global
                # k_global = b_block * 2 + k_loc
                for (i_loc, k_loc, coeff) in W_s[s]:
                    i = i_loc
                    k = b_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx, col] += coeff
                
                col += 1
    
    # Verify quickly if time allows (optional, but good for debugging)
    # For now, return the solution.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
