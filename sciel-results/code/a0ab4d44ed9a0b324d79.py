"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10  # 10 seconds is plenty for this O(1) construction
    
    n, m, p = DIMENSIONS
    # Rank 56 using Strassen for each 2x2 block of B
    # 8 blocks, 7 terms each
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(8):
        if time.time() > deadline:
            break
            
        # k is the block index (0 to 7)
        # B block columns are 2k and 2k+1
        c0 = 2 * k
        c1 = 2 * k + 1
        
        # Strassen Term 1: M1 = (a00+a11)(b00+b11)
        U[0][col] = 1  # i=0, j=0
        U[3][col] = 1  # i=1, j=1
        V[0 * p + c0][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # Strassen Term 2: M2 = (a10+a11)b00
        U[2][col] = 1  # i=1, j=0
        U[3][col] = 1  # i=1, j=1
        V[0 * p + c0][col] = 1
        W[c0 * n + 1][col] = 1
        col += 1
        
        # Strassen Term 3: M3 = a00(b01-b11)
        U[0][col] = 1  # i=0, j=0
        V[0 * p + c1][col] = 1
        V[1 * p + c1][col] = -1
        W[c1 * n + 0][col] = 1
        col += 1
        
        # Strassen Term 4: M4 = a11(b10-b00)
        U[3][col] = 1  # i=1, j=1
        V[1 * p + c0][col] = 1
        V[0 * p + c0][col] = -1
        W[c0 * n + 0][col] = 1
        W[c0 * n + 1][col] = 1
        col += 1
        
        # Strassen Term 5: M5 = (a00+a01)b11
        U[0][col] = 1  # i=0, j=0
        U[1][col] = 1  # i=0, j=1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = -1
        W[c1 * n + 0][col] = 1
        col += 1
        
        # Strassen Term 6: M6 = (a10-a00)(b00+b01)
        U[2][col] = 1  # i=1, j=0
        U[0][col] = -1 # i=0, j=0
        V[0 * p + c0][col] = 1
        V[0 * p + c1][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # Strassen Term 7: M7 = (a01-a11)(b10+b11)
        U[1][col] = 1  # i=0, j=1
        U[3][col] = -1 # i=1, j=1
        V[1 * p + c0][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
