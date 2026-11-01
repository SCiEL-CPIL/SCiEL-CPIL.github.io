"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: 2x1x6 (j=0)
    col = 0
    for i in range(2):
        for k in range(6):
            U[5*i + 0][col] = 1
            V[0*6 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1

    def add_strassen_block(j_offset, k_offset, col_start):
        U[5*0 + j_offset + 0][col_start] += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        col_start += 1
        U[5*1 + j_offset + 0][col_start] += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        W[(k_offset+0)*2 + 1][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] -= 1
        
        col_start += 1
        U[5*0 + j_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] -= 1
        W[(k_offset+1)*2 + 0][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        col_start += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] -= 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        W[(k_offset+0)*2 + 1][col_start] += 1
        
        col_start += 1
        U[5*0 + j_offset + 0][col_start] += 1
        U[5*0 + j_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] -= 1
        W[(k_offset+1)*2 + 0][col_start] += 1
        
        col_start += 1
        U[5*1 + j_offset + 0][col_start] += 1
        U[5*0 + j_offset + 0][col_start] -= 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        col_start += 1
        U[5*0 + j_offset + 1][col_start] += 1
        U[5*1 + j_offset + 1][col_start] -= 1
        V[(j_offset+1)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        
        return col_start + 1

    col = add_strassen_block(1, 0, 12)
    col = add_strassen_block(1, 2, col)
    col = add_strassen_block(1, 4, col)
    col = add_strassen_block(3, 0, col)
    col = add_strassen_block(3, 2, col)
    col = add_strassen_block(3, 4, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
