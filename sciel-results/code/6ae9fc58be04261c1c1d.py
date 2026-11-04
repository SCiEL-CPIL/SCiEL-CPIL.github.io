"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 28 decomposition using 4 Strassen blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication
    # Local indices: 0:a00, 1:a01, 2:a10, 3:a11
    strassen_terms = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)), # M1
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)), # M2
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)), # M3
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)), # M4
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)), # M5
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)), # M6
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0))  # M7
    ]
    
    # Block definitions: (A_indices, B_indices, C_indices)
    # A is 2x4 (8 elements), B is 4x4 (16 elements), C is 2x4 (8 elements)
    # Indices are 0-based flat row-major
    
    # Block 1: A_L (cols 0,1), B_LL (rows 0,1 cols 0,1) -> C_L (cols 0,1)
    # A_L indices: 0,1,2,3
    # B_LL indices: 0,1,4,5
    # C_L indices: 0,1,2,3
    blocks = [
        ([0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 2, 3]),
        # Block 2: A_R (cols 2,3), B_RL (rows 2,3 cols 0,1) -> C_L (cols 0,1)
        # A_R indices: 4,5,6,7
        # B_RL indices: 8,9,12,13
        # C_L indices: 0,1,2,3
        ([4, 5, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3]),
        # Block 3: A_L (cols 0,1), B_LR (rows 0,1 cols 2,3) -> C_R (cols 2,3)
        # A_L indices: 0,1,2,3
        # B_LR indices: 2,3,6,7
        # C_R indices: 4,5,6,7
        ([0, 1, 2, 3], [2, 3, 6, 7], [4, 5, 6, 7]),
        # Block 4: A_R (cols 2,3), B_RR (rows 2,3 cols 2,3) -> C_R (cols 2,3)
        # A_R indices: 4,5,6,7
        # B_RR indices: 10,11,14,15
        # C_R indices: 4,5,6,7
        ([4, 5, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7])
    ]
    
    col = 0
    for a_idx, b_idx, c_idx in blocks:
        for u_loc, v_loc, w_loc in strassen_terms:
            for k in range(4):
                if u_loc[k] != 0:
                    U[a_idx[k]][col] += u_loc[k]
                if v_loc[k] != 0:
                    V[b_idx[k]][col] += v_loc[k]
                if w_loc[k] != 0:
                    W[c_idx[k]][col] += w_loc[k]
            col += 1
            if time.time() - start_time > 145:
                break
        if time.time() - start_time > 145:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
