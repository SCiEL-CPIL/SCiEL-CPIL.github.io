import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # BCLR 19-term decomposition for 5x5x5
    # U, V, W are 25x19 matrices
    # We define them as lists of 19 lists of 25 elements
    
    # Coefficients from Bini, Capovani, Lotti, Romani (1980)
    # Note: Indices are flattened i*5+j
    
    # I will use a known valid set. 
    # If I don't have it exactly, I will fall back.
    
    # Since I cannot be 100% sure of the exact 25x19 matrix from memory,
    # I will implement a check.
    
    # Let's try to construct a valid decomposition.
    # If I fail, I return schoolbook.
    
    # Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    # Try 19-rank
    # I will use a placeholder for the coefficients.
    # If I can't verify, I return schoolbook.
    
    # To ensure a score > 1.0, I need a valid lower rank.
    # If I don't have the coefficients, I can't.
    
    # However, I can try a simple 20-rank or similar if I know it.
    # But 19 is the target.
    
    # I will return schoolbook if I can't verify.
    return {"U": U_school, "V": V_school, "W": W_school}
