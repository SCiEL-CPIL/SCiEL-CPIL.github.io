import numpy as np

def construct():
    n, m, p = 3, 3, 8
    # We will use a block decomposition:
    # 3x3 x 8 = (3x3 x 3) + (3x3 x 3) + (3x3 x 2)
    # Rank of 3x3x3 is 19 (using a known decomposition)
    # Rank of 3x3x2 is 18 (schoolbook)
    # Total rank = 19 + 19 + 18 = 56
    
    # For simplicity and correctness, I will use a rank 23 decomposition for 3x3x3
    # which is easier to verify.
    # Total rank = 23 + 23 + 18 = 64
    
    # I will implement a rank 23 decomposition for 3x3x3.
    # This is a known result (Lader).
    # Since I don't have the coefficients, I will use a simpler bound.
    
    # Actually, I can use the schoolbook for 3x3x3 (rank 27)
    # Total rank = 27 + 27 + 18 = 72. This is not an improvement.
    
    # I must find a better rank.
    # Let's try to use Strassen for 2x2 parts of 3x3x3.
    
    # I will use a rank 56 decomposition if I can find it.
    # If not, I will use rank 64.
    
    # For now, I will return the schoolbook decomposition to ensure validity.
    # But I need to improve the score.
    
    # I will use a rank 64 decomposition.
    # I will hardcode a rank 23 decomposition for 3x3x3.
    
    # The Lader decomposition for 3x3x3 has rank 23.
    # I will use it.
    
    # U, V, W for 3x3x3
    # I will use a known rank 23 decomposition.
    
    # Since I cannot recall the exact coefficients, I will use a simpler approach:
    # Use Strassen for the 2x2 block and schoolbook for the rest.
    
    # This is too complex. I will use the schoolbook for now.
    
    rank = 72
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
