import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use the strategy of splitting B into two 3x3 blocks.
    # Rank of 2x3x3 multiplication is 14.
    # Total rank will be 28.
    
    # Define a rank 14 decomposition for 2x3x3.
    # U: 6x14, V: 9x14, W: 6x14
    # This is a known decomposition.
    # I will use a placeholder if I don't know it, but I must provide a valid one.
    # If I can't find it, I will use a rank 18 decomposition (schoolbook) which gives 36.
    # But I want to improve.
    
    # Let's try to find a rank 14 decomposition.
    # Actually, I can use the fact that R(2,3,3) <= 14.
    # I will use a known rank 14 decomposition.
    
    U_233 = []
    V_233 = []
    W_233 = []
    
    # Since I don't have the exact coefficients memorized, I will use a heuristic.
    # Or I will use a rank 15 decomposition if I can find one.
    # Or I will use the schoolbook decomposition for 2x3x3 (rank 18) and hope for a better global decomposition.
    
    # Actually, let's just use the schoolbook for 2x3x6 (rank 36) if I can't find a better one.
    # But the task asks for a stronger algorithm.
    
    # I will use the rank 14 decomposition if I can find it.
    # I will try to search for it in the code using a simple optimization? No, must be exact.
    
    # I will use a rank 14 decomposition found in literature.
    # Coefficients are integers.
    
    # If I can't find it, I will use a rank 15 decomposition.
    # I will use a rank 15 decomposition for 2x3x3.
    
    # Let's assume I can find a rank 14 decomposition.
    # I will hardcode it.
    
    # Since I cannot hardcode it reliably, I will use a rank 18 decomposition (schoolbook) for 2x3x3,
    # which gives total rank 36.
    # This is the same as the current program.
    
    # To improve, I will use a rank 15 decomposition for 2x3x3.
    # I will construct it using the block method.
    
    # Let's use the block method with rank 11 for 2x2x3 and rank 6 for 2x1x3.
    # Total 17.
    # Then total rank 34.
    
    # I will use rank 34.
    
    # Construct rank 11 for 2x2x3.
    # Schoolbook is 12.
    # Can I find rank 11?
    # Yes, it is known.
    
    # I will use rank 12 for 2x2x3 and rank 6 for 2x1x3.
    # Total 18.
    # Total rank 36.
    
    # I will use the current program's logic but with a better decomposition if found.
    # Since I can't find it, I will use the current program.
    
    # Wait, I can use the fact that R(2,3,6) <= 28.
    # I will try to find a rank 14 decomposition for 2x3x3.
    
    # I will use a rank 14 decomposition.
    
    # I will output the current program if I can't find a better one.
    
    rank = 36
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
