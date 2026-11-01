"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline to be safe
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1
        U[a00_u][offset + 0] += 1
        U[a11_u][offset + 0] += 1
        V[b00_v][offset + 0] += 1
        V[b11_v][offset + 0] += 1
        W[c00_w][offset + 0] += 1
        W[c11_w][offset + 0] += 1
        
        # M2
        U[a10_u][offset + 1] += 1
        U[a11_u][offset + 1] += 1
        V[b00_v][offset + 1] += 1
        W[c10_w][offset + 1] += 1
        W[c11_w][offset + 1] -= 1
        
        # M3
        U[a00_u][offset + 2] += 1
        V[b01_v][offset + 2] += 1
        V[b11_v][offset + 2] -= 1
        W[c01_w][offset + 2] += 1
        W[c11_w][offset + 2] += 1
        
        # M4
        U[a11_u][offset + 3] += 1
        V[b10_v][offset + 3] += 1
        V[b00_v][offset + 3] -= 1
        W[c00_w][offset + 3] += 1
        W[c10_w][offset + 3] += 1
        
        # M5
        U[a00_u][offset + 4] += 1
        U[a01_u][offset + 4] += 1
        V[b11_v][offset + 4] += 1
        W[c00_w][offset + 4] -= 1
        W[c01_w][offset + 4] += 1
        
        # M6
        U[a10_u][offset + 5] += 1
        U[a00_u][offset + 5] -= 1
        V[b00_v][offset + 5] += 1
        V[b01_v][offset + 5] += 1
        W[c11_w][offset + 5] += 1
        
        # M7
        U[a01_u][offset + 6] += 1
        U[a11_u][offset + 6] -= 1
        V[b10_v][offset + 6] += 1
        V[b11_v][offset + 6] += 1
        W[c00_w][offset + 6] += 1

    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    if time.time() > deadline:
        pass # Just in case, though construction is fast
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
