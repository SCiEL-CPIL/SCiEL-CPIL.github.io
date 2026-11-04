"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s margin
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Splitting: 4×4×5 = 4×(2×2×2) ⊕ 4×(2×2×3)
    # rank ≤ 4*7 + 4*r(2,2,3)
    
    # For now, use schoolbook for 2×2×3 (rank 12)
    # Total rank = 4*7 + 4*12 = 28 + 48 = 76
    
    # But let's try to find a better decomposition for 2×2×3
    
    # Schoolbook decomposition for 2×2×2 (rank 8, but we know rank 7 exists)
    # For simplicity, let's use schoolbook for everything first
    
    # Actually, let me just use the schoolbook for the full 4×4×5 (rank 80)
    # and see if I can improve it.
    
    # Let me try the splitting approach with schoolbook sub-decompositions.
    
    # 4×4×5 split into 8 blocks: 4 of type 2×2×2 and 4 of type 2×2×3
    # Using schoolbook: 4*8 + 4*12 = 32 + 48 = 80 (same as before!)
    
    # So I need to use rank-7 for 2×2×2 and rank < 12 for 2×2×3.
    
    # Let me use rank-7 for 2×2×2 (Strassen) and rank-12 for 2×2×3 (schoolbook)
    # Total: 4*7 + 4*12 = 28 + 48 = 76
    
    # Or rank-7 for 2×2×2 and rank-10 for 2×2×3:
    # Total: 4*7 + 4*10 = 28 + 40 = 68
    
    # Let me try to find a rank-10 decomposition for 2×2×3.
    
    # For now, let me just return the schoolbook decomposition (rank 80)
    # and add optimization later.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
