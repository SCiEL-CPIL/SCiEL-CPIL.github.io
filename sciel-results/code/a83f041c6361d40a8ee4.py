import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split B (4x6) into B1 (4x4) and B2 (4x2)
    # C = A * B = A * B1 + A * B2
    # C1 = A * B1 (4x4), C2 = A * B2 (4x2)
    
    # Rank for 4x4x4 using Laderman's algorithm is 49
    # Rank for 4x4x2 using Schoolbook is 32
    rank_laderman = 49
    rank_schoolbook = 32
    total_rank = rank_laderman + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 (Laderman) ---
    # Indices for B1: cols 0-3
    # U, V, W for Laderman
    # Note: I need the actual Laderman matrices. 
    # Since I cannot guarantee the exact 49 coefficients from memory, 
    # I will use a fallback: if I were to use schoolbook for 4x4x4, rank is 64.
    # 64 + 32 = 96. No improvement.
    # I will stick to schoolbook for the entire 4x4x6 to ensure correctness, 
    # as I cannot reliably generate Laderman's specific factors here.
    
    # Reverting to Schoolbook for safety
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
