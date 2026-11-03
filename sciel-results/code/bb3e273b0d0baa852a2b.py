import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
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
    
    # Try Rank 46 decomposition (2 * Rank 23 for 3x3x3)
    # Rank 23 decomposition of 3x3x3 matrix multiplication
    # U, V, W are 9x23 matrices
    # We need the coefficients.
    # I will use a known Rank 23 decomposition.
    # If I don't have it, I will skip.
    # For now, I will assume I can construct it.
    # Actually, I will use a placeholder Rank 23 decomposition.
    # If I cannot, I will return schoolbook.
    
    # Let's try to implement Rank 23.
    # The Rank 23 decomposition is:
    # ... (coefficients) ...
    
    # Since I don't have the exact coefficients memorized, 
    # I will return the schoolbook decomposition to ensure validity.
    # However, I can try to use a Rank 28 decomposition? No, 56 > 54.
    # So Rank 46 is the target.
    
    # I will return schoolbook for now, as I cannot guarantee the Rank 23 coefficients.
    # Wait, I can use a Rank 23 decomposition from a library? No.
    
    # I will return schoolbook.
    return {"U": U_sb, "V": V_sb, "W": W_sb}
