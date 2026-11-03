import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook Rank 63
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    # Attempt to use Laderman (Rank 23) for 3x3 blocks
    # This is risky if matrices are wrong.
    # I will use a known correct Laderman implementation.
    
    # Laderman 3x3x3
    U_l = []
    V_l = []
    W_l = []
    
    # ... (I will skip the actual Laderman definitions if I can't recall them reliably)
    
    # To ensure a score > 1.0, I will use a decomposition that I am sure of.
    # Is there one?
    # Rank of 3x3 is 23.
    # I will use the fact that 23 < 27.
    # I will output the Schoolbook for now, as I cannot recall Laderman.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
