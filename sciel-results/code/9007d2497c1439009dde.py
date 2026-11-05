import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s limit to be safe
    
    # Function to build schoolbook decomposition for a block
    def schoolbook_block(n_b, m_b, p_b):
        rank = n_b * m_b * p_b
        U = [[0] * rank for _ in range(n_b * m_b)]
        V = [[0] * rank for _ in range(m_b * p_b)]
        W = [[0] * rank for _ in range(p_b * n_b)]
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i * m_b + j][col] = 1
                    V[j * p_b + k][col] = 1
                    W[k * n_b + i][col] = 1
                    col += 1
        return U, V, W, rank

    # Attempt to use a rank-23 decomposition for 3x3x3
    # If we don't have the exact coefficients, we use schoolbook (rank 27)
    # Here I will use schoolbook for safety, but structure the code to allow replacement
    # For now, let's just use schoolbook for all blocks to ensure validity.
    # To get an improvement, I need the 23-term coefficients.
    # Since I don't have them, I will return the schoolbook decomposition.
    
    U, V, W, rank = schoolbook_block(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
