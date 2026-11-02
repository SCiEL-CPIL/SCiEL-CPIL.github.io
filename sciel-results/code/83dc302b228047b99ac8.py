import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank minimal decomposition using Strassen's algorithm for two 2x2x2 blocks
    n, m, p = DIMENSIONS
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # block_offset: column offset in U, V, W
    # v_offset: row offset in V (0 for first block, 2 for second block? No, V rows are j*p+k)
    # Let's map indices explicitly.
    
    # Block 1: k in {0, 1}
    # V indices: e=0, f=1, g=4, h=5
    # W indices: C00=0, C10=1, C01=2, C11=3
    
    # Block 2: k in {2, 3}
    # V indices: e=2, f=3, g=6, h=7
    # W indices: C02=4, C12=5, C03=6, C13=7
    
    def add_strassen_block(col_offset, v_e, v_f, v_g, v_h, w_p, w_q, w_r, w_s):
        # M1 = (a+d)(e+h)
        U[0][col_offset] += 1
        U[3][col_offset] += 1
        V[v_e][col_offset] += 1
        V[v_h][col_offset] += 1
        W[w_p][col_offset] += 1
        W[w_s][col_offset] += 1
        
        # M2 = (c+d)e
        U[2][col_offset+1] += 1
        U[3][col_offset+1] += 1
        V[v_e][col_offset+1] += 1
        W[w_r][col_offset+1] += 1
        W[w_s][col_offset+1] -= 1
        
        # M3 = a(f-h)
        U[0][col_offset+2] += 1
        V[v_f][col_offset+2] += 1
        V[v_h][col_offset+2] -= 1
        W[w_q][col_offset+2] += 1
        W[w_s][col_offset+2] += 1
        
        # M4 = d(g-e)
        U[3][col_offset+3] += 1
        V[v_g][col_offset+3] += 1
        V[v_e][col_offset+3] -= 1
        W[w_p][col_offset+3] += 1
        W[w_r][col_offset+3] += 1
        
        # M5 = (a+b)h
        U[0][col_offset+4] += 1
        U[1][col_offset+4] += 1
        V[v_h][col_offset+4] += 1
        W[w_p][col_offset+4] -= 1
        W[w_q][col_offset+4] += 1
        
        # M6 = (c-a)(e+f)
        U[2][col_offset+5] += 1
        U[0][col_offset+5] -= 1
        V[v_e][col_offset+5] += 1
        V[v_f][col_offset+5] += 1
        W[w_s][col_offset+5] += 1
        
        # M7 = (b-d)(g+h)
        U[1][col_offset+6] += 1
        U[3][col_offset+6] -= 1
        V[v_g][col_offset+6] += 1
        V[v_h][col_offset+6] += 1
        W[w_p][col_offset+6] += 1

    # Block 1
    # v_e=0, v_f=1, v_g=4, v_h=5
    # w_p=0 (C00), w_q=2 (C01), w_r=1 (C10), w_s=3 (C11)
    add_strassen_block(0, 0, 1, 4, 5, 0, 2, 1, 3)
    
    # Block 2
    # v_e=2, v_f=3, v_g=6, v_h=7
    # w_p=4 (C02), w_q=6 (C03), w_r=5 (C12), w_s=7 (C13)
    add_strassen_block(7, 2, 3, 6, 7, 4, 6, 5, 7)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
