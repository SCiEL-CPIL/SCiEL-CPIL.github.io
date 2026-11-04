"""Rewritten program for matrix multiplication 2x4x4 using Strassen decomposition."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 * 2x2 -> 2x2
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X10)Y11
    # M6 = (X11-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Blocks: (j_start, k_start)
    # j_start in {0, 2}
    # k_start in {0, 2}
    blocks = [
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2)
    ]
    
    col_offset = 0
    for j_start, k_start in blocks:
        if time.time() > deadline:
            break
            
        # M1
        U[0*m + (j_start+0)][col_offset] += 1
        U[1*m + (j_start+1)][col_offset] += 1
        V[(j_start+0)*p + (k_start+0)][col_offset] += 1
        V[(j_start+1)*p + (k_start+1)][col_offset] += 1
        W[(k_start+0)*n + 0][col_offset] += 1
        W[(k_start+1)*n + 1][col_offset] += 1
        col_offset += 1
        
        # M2
        U[1*m + (j_start+0)][col_offset] += 1
        U[1*m + (j_start+1)][col_offset] += 1
        V[(j_start+0)*p + (k_start+0)][col_offset] += 1
        W[(k_start+0)*n + 1][col_offset] += 1
        col_offset += 1
        
        # M3
        U[0*m + (j_start+0)][col_offset] += 1
        V[(j_start+1)*p + (k_start+0)][col_offset] += 1
        V[(j_start+1)*p + (k_start+1)][col_offset] -= 1
        W[(k_start+1)*n + 0][col_offset] += 1
        col_offset += 1
        
        # M4
        U[1*m + (j_start+1)][col_offset] += 1
        V[(j_start+0)*p + (k_start+1)][col_offset] += 1
        V[(j_start+0)*p + (k_start+0)][col_offset] -= 1
        W[(k_start+0)*n + 0][col_offset] += 1
        W[(k_start+0)*n + 1][col_offset] += 1
        col_offset += 1
        
        # M5
        U[0*m + (j_start+0)][col_offset] += 1
        U[1*m + (j_start+0)][col_offset] += 1
        V[(j_start+1)*p + (k_start+1)][col_offset] += 1
        W[(k_start+0)*n + 0][col_offset] -= 1
        W[(k_start+1)*n + 0][col_offset] += 1
        col_offset += 1
        
        # M6
        U[1*m + (j_start+1)][col_offset] += 1
        U[0*m + (j_start+0)][col_offset] -= 1
        V[(j_start+0)*p + (k_start+0)][col_offset] += 1
        V[(j_start+0)*p + (k_start+1)][col_offset] += 1
        W[(k_start+1)*n + 1][col_offset] += 1
        col_offset += 1
        
        # M7
        U[0*m + (j_start+1)][col_offset] += 1
        U[1*m + (j_start+1)][col_offset] -= 1
        V[(j_start+1)*p + (k_start+0)][col_offset] += 1
        V[(j_start+1)*p + (k_start+1)][col_offset] += 1
        W[(k_start+0)*n + 0][col_offset] += 1
        col_offset += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
