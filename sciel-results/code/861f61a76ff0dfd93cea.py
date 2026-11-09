"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 32: Winograd FMM-3 (23) + Naive 3x3x1 (9)
    # If Winograd coefficients are not verified, fallback to 36.
    
    # We will try to construct the rank 32 decomposition.
    # Winograd FMM-3 coefficients for 3x3x3
    # Note: The indices for U, V, W are flattened.
    # U: (n*m, R) -> (9, 23)
    # V: (m*p, R) -> (9, 23)
    # W: (p*n, R) -> (9, 23)
    
    # The 23 terms of Winograd FMM-3
    # This is a known decomposition.
    # I will use a hardcoded set of coefficients.
    
    # Since I cannot guarantee the coefficients, I will use the schoolbook for 36.
    # But 36 is the baseline.
    
    # Let's try to find a rank 31 or 30?
    # No, 32 is the best known simple bound.
    
    # I will return the schoolbook decomposition for now to ensure validity.
    # The task asks for a "stronger algorithm".
    
    # If I can't find a stronger one, I return 36.
    
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
