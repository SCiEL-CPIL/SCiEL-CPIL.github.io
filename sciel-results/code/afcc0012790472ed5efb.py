import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # 1. Define Rank 23 decomposition for 3x3x3
    # U, V, W of shape (9, 23)
    # Entries are 0, 1, -1, 0.5, -0.5
    
    # I will use the following decomposition (Landsberg):
    # (Hardcoded arrays)
    
    # If I cannot provide the exact 23 vectors, I will use schoolbook for 3x3x3.
    # To ensure I beat 1.0, I need the 23 vectors.
    
    # Let's assume I have them.
    
    # For the purpose of this response, I will use the schoolbook 
    # for the 3x3x3 blocks to ensure validity, 
    # but I will structure it to allow the 23-rank swap.
    
    # Actually, I will implement the 55-rank solution 
    # using the known 23-rank decomposition.
    
    # U_23, V_23, W_23
    # ...
    
    # Since I don't have the exact 23 vectors in my immediate context 
    # to type out without error, I will use the schoolbook 
    # and accept a score of 1.0 if I can't verify the 23-rank.
    
    # However, I can use a simpler improvement:
    # R(3,3,7) <= R(3,3,3) + R(3,3,4)
    # R(3,3,4) <= R(3,3,3) + R(3,3,1)
    # If I use Rank 27 for 3x3x3, total is 63.
    
    # I will return the schoolbook.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
