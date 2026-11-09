import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    rank = 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen matrices for 2x2x2
    # U_strassen: 4x7
    # Rows: A00, A01, A10, A11
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    # V_strassen: 4x7
    # Rows: B00, B01, B10, B11
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    # W_strassen: 4x7
    # Rows: C00, C01, C10, C11
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Helper to add Strassen block
    def add_strassen_block(u_offset, v_offset, w_offset, rank_offset):
        # u_offset: row offset in U for A00 (i=0, j=0)
        # v_offset: row offset in V for B00 (j=0, k=0)
        # w_offset: row offset in W for C00 (k=0, i=0)
        
        # Mapping Strassen indices (0..3) to Full indices
        # A: 00->0, 01->1, 10->3, 11->4 (relative to u_offset base? No, u_offset is row of A00)
        # A00 is at u_offset
        # A01 is at u_offset + 1
        # A10 is at u_offset + 3 (since j=0 is col 0, j=1 is col 1. i=1 is row 1. 1*3+0=3)
        # A11 is at u_offset + 4
        
        u_map = [u_offset, u_offset + 1, u_offset + 3, u_offset + 4]
        
        # B: 00->0, 01->1, 10->4, 11->5 (relative to v_offset base)
        # B00 is at v_offset
        # B01 is at v_offset + 1
        # B10 is at v_offset + 4 (j=1, k=0. 1*4+0=4)
        # B11 is at v_offset + 5
        
        v_map = [v_offset, v_offset + 1, v_offset + 4, v_offset + 5]
        
        # W: 00->0, 01->1, 10->2, 11->3 (relative to w_offset base)
        # C00 is at w_offset
        # C01 is at w_offset + 1
        # C10 is at w_offset + 2 (k=1, i=0. 1*2+0=2)
        # C11 is at w_offset + 3
        
        w_map = [w_offset, w_offset + 1, w_offset + 2, w_offset + 3]
        
        for r in range(7):
            col = rank_offset + r
            for s_idx in range(4):
                U[u_map[s_idx]][col] = U_s[s_idx][r]
                V[v_map[s_idx]][col] = V_s[s_idx][r]
                W[w_map[s_idx]][col] = W_s[s_idx][r]
                
    # Block 1: A (j=0,1), B (j=0,1; k=0,1), C (k=0,1)
    # A00 at U[0,0] -> u_offset = 0
    # B00 at V[0,0] -> v_offset = 0
    # C00 at W[0,0] -> w_offset = 0
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A (j=0,1), B (j=0,1; k=2,3), C (k=2,3)
    # A00 at U[0,0] -> u_offset = 0
    # B00 at V[0,2] -> v_offset = 2
    # C00 at W[2,0] -> w_offset = 4 (k=2, i=0 -> 2*2+0=4)
    add_strassen_block(0, 2, 4, 7)
    
    # Block 3: A (j=2), B (j=2; k=0..3), C (k=0..3)
    # Rank 1
    col = 14
    # A: i=0, j=2 -> U[2, 14] = 1
    #     i=1, j=2 -> U[5, 14] = 1
    U[2][col] = 1
    U[5][col] = 1
    
    # B: j=2, k=0..3 -> V[8+k, 14] = 1
    for k in range(4):
        V[2*4 + k][col] = 1
        
    # C: k=0..3, i=0..1 -> W[k*2+i, 14] = 1
    for k in range(4):
        for i in range(2):
            W[k*2 + i][col] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
