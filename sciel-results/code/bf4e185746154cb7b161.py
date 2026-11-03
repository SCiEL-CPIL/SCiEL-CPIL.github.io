import time
import sys
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    t_limit = 115.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    
    # Base decomposition for 2x2 matrix multiplication (rank 7)
    # Local indices:
    # A: 2x2 -> flat 0..3 (i*2+j)
    # B: 2x2 -> flat 0..3 (j*2+k)
    # C: 2x2 -> flat 0..3 (i*2+k) -- NOTE: Local C flat index for mapping to W
    # But W uses convention k*n+i. We handle mapping in global step.
    
    # Strassen components
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1)
    U_local[0][0] = 1; U_local[3][0] = 1 # a00, a11
    V_local[0][0] = 1; V_local[3][0] = 1 # b00, b11
    W_local[0][0] = 1 # C00
    W_local[3][0] = 1 # C11
    
    # r=1 (M2)
    U_local[2][1] = 1; U_local[3][1] = 1 # a10, a11
    V_local[0][1] = 1 # b00
    W_local[2][1] = 1 # C10
    W_local[3][1] = -1 # C11
    
    # r=2 (M3)
    U_local[0][2] = 1 # a00
    V_local[1][2] = 1; V_local[3][2] = -1 # b01, b11
    W_local[1][2] = 1 # C01
    W_local[3][2] = 1 # C11
    
    # r=3 (M4)
    U_local[3][3] = 1 # a11
    V_local[2][3] = 1; V_local[0][3] = -1 # b10, b00
    W_local[0][3] = 1 # C00
    W_local[2][3] = 1 # C10
    
    # r=4 (M5)
    U_local[0][4] = 1; U_local[1][4] = 1 # a00, a01
    V_local[3][4] = 1 # b11
    W_local[0][4] = -1 # C00
    W_local[1][4] = 1 # C01
    
    # r=5 (M6)
    U_local[2][5] = 1; U_local[0][5] = -1 # a10, a00
    V_local[0][5] = 1; V_local[1][5] = 1 # b00, b01
    W_local[3][5] = 1 # C11
    
    # r=6 (M7)
    U_local[1][6] = 1; U_local[3][6] = -1 # a01, a11
    V_local[2][6] = 1; V_local[3][6] = 1 # b10, b11
    W_local[0][6] = 1 # C00

    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)] # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)] # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)] # 16 x 56
    
    # Mapping logic
    # j: 0..3 (col block of C)
    # s: 0..1 (selector for A row-block/B row-block)
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # Define mappings for local indices 0..3 to global indices
            # Local A: i_loc (0,1), j_loc (0,1) -> flat i_loc*2+j_loc
            # Global A: i_g (0,1), j_g (0,1,2,3) -> flat i_g*4+j_g
            a_map = [0]*4
            if s == 0:
                # A1: cols 0,1
                a_map[0] = 0*4 + 0 # i=0, j=0
                a_map[1] = 0*4 + 1 # i=0, j=1
                a_map[2] = 1*4 + 0 # i=1, j=0
                a_map[3] = 1*4 + 1 # i=1, j=1
            else:
                # A2: cols 2,3
                a_map[0] = 0*4 + 2
                a_map[1] = 0*4 + 3
                a_map[2] = 1*4 + 2
                a_map[3] = 1*4 + 3
                
            # Local B: j_loc (0,1), k_loc (0,1) -> flat j_loc*2+k_loc
            # Global B: j_g (0..3), k_g (0..7) -> flat j_g*8+k_g
            b_map = [0]*4
            if s == 0:
                # Top half (rows 0,1), cols 2j, 2j+1
                b_map[0] = 0*8 + (2*j)   # j_g=0, k_g=2j
                b_map[1] = 0*8 + (2*j+1) # j_g=0, k_g=2j+1
                b_map[2] = 1*8 + (2*j)   # j_g=1, k_g=2j
                b_map[3] = 1*8 + (2*j+1) # j_g=1, k_g=2j+1
            else:
                # Bottom half (rows 2,3), cols 2j, 2j+1
                b_map[0] = 2*8 + (2*j)
                b_map[1] = 2*8 + (2*j+1)
                b_map[2] = 3*8 + (2*j)
                b_map[3] = 3*8 + (2*j+1)
                
            # Local C: i_loc (0,1), k_loc (0,1) -> flat i_loc*2+k_loc
            # Global C: i_g (0,1), k_g (0..7)
            # W Index Convention: k_g * n + i_g = k_g * 2 + i_g
            c_map = [0]*4
            # k_g is 2j or 2j+1
            # i_g is 0 or 1
            
            # Local C index 0: i=0, k_loc=0 -> k_g=2j
            c_map[0] = (2*j) * 2 + 0
            # Local C index 1: i=0, k_loc=1 -> k_g=2j+1
            c_map[1] = (2*j+1) * 2 + 0
            # Local C index 2: i=1, k_loc=0 -> k_g=2j
            c_map[2] = (2*j) * 2 + 1
            # Local C index 3: i=1, k_loc=1 -> k_g=2j+1
            c_map[3] = (2*j+1) * 2 + 1

            # Fill U, V, W
            for l in range(7):
                r = base_r + l
                
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        ga = a_map[la]
                        U[ga][r] += val
                        
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        gb = b_map[lb]
                        V[gb][r] += val
                        
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        gc = c_map[lc]
                        W[gc][r] += val
                        
    if time.time() - start_time > t_limit:
        # Safety check, though this is deterministic and fast
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
