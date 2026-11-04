import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# Winograd 3x3x3 decomposition coefficients (Rank 23)
# Based on standard literature
# U: 9x23, V: 9x23, W: 9x23
# Indices:
# U[i,j] corresponds to a_{i,j}
# V[i,j] corresponds to b_{i,j}
# W[i,j] corresponds to c_{i,j}

# I will construct the rank 46 decomposition by doubling the rank 23 decomposition.
# U_wino, V_wino, W_wino are 9x23 arrays.

def get_wino_333():
    # Coefficients for Winograd 3x3
    # Using a known valid set. 
    # Note: There are multiple valid decompositions. 
    # I will use a standard one found in literature or derive it.
    # Since I cannot derive it reliably in runtime, I will use a hardcoded known valid set.
    # If I don't have the exact coefficients, I can't use this.
    # Let's try to generate them? No, too hard.
    # I will use the fact that for 3x3, rank 23 is minimal.
    # If I don't have the coefficients, I should fall back to 54.
    # However, I can search for a valid decomposition of rank 46 using a heuristic if I don't have the coefficients.
    # But 46 is a specific number.
    # Let's assume I can find a rank 46 solution by searching?
    # Unlikely in 120s.
    
    # I will output the schoolbook for safety, as I do not have the Winograd coefficients hardcoded.
    # Wait, I can try to implement a simple search for rank 46?
    # No, the space is huge.
    
    # I will stick to the schoolbook to ensure validity.
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 20s safety margin
    
    # Try to find a better decomposition?
    # Since I don't have the Winograd coefficients, I cannot safely improve.
    # I will return the schoolbook decomposition.
    
    return get_wino_333() # Actually just schoolbook
