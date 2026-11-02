"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Terms M1..M7
    # A_coeffs: list of 7 lists of (i, k, coeff)
    # B_coeffs: list of 7 lists of (k, j, coeff)
    # C_coeffs: list of 7 lists of (i, j, coeff)
    
    A_coeffs = [
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1), (1,1,1)],       # M2
        [(0,0,1)],                # M3
        [(1,1,1)],                # M4
        [(0,0,1), (0,1,1)],       # M5
        [(1,0,1), (0,0,-1)],      # M6
        [(0,1,1), (1,1,-1)]       # M7
    ]
    
    B_coeffs = [
        [(0,0,1), (1,1,1)],       # M1
        [(0,0,1)],                # M2
        [(0,1,1), (1,1,-1)],      # M3
        [(1,0,1), (0,0,-1)],      # M4
        [(1,1,1)],                # M5
        [(0,0,1), (0,1,1)],       # M6
        [(1,0,1), (1,1,1)]        # M7
    ]
    
    C_coeffs = [
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1)],                # M2
        [(0,1,1)],                # M3
        [(0,0,1), (1,0,1)],       # M4
        [(0,0,-1), (0,1,1)],      # M5
        [(1,1,1)],                # M6
        [(0,0,1), (1,1,1)]        # M7
    ]
    
    # Define the 4 blocks
    # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 0),
        (2, 2, 0, 0),
        (2, 2, 2, 2)
    ]
    
    col_start = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for t in range(7):
            col = col_start + t
            
            # Update U
            for i, k, coeff in A_coeffs[t]:
                u_row = i * m + (a_off + k)
                U[u_row][col] += coeff
                
            # Update V
            for k, j, coeff in B_coeffs[t]:
                v_row = (b_row_off + k) * p + (b_col_off + j)
                V[v_row][col] += coeff
                
            # Update W
            for i, j, coeff in C_coeffs[t]:
                w_row = (c_off + j) * n + i
                W[w_row][col] += coeff
                
        col_start += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
