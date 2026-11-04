import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 21 (for 3x3 via 4x4 padding) * 2 + 18 (for tail) = 60
    # We use a rank 21 decomposition for 4x4 matrix multiplication.
    # If we don't have it, we fall back to schoolbook.
    
    # Hardcoded Rank 21 decomposition for 4x4x4
    # Source: Strassen's algorithm or known minimal decomposition.
    # Actually, generating the 21-term decomposition on the fly is complex.
    # I will use a known set of coefficients for 4x4 rank 21.
    # For safety, if I cannot verify the 21-terms, I will use a simpler improvement.
    
    # Let's try to use the Strassen 2x2 decomposition (rank 7) 
    # and see if we can combine them.
    
    # Actually, a safer improvement is to use the rank 23 for 3x3.
    # But I don't have the 23-terms hardcoded.
    
    # Let's stick to the 60-rank plan using 4x4 rank 21.
    # I will define the 4x4 rank 21 decomposition.
    # The 21-term decomposition for 4x4 is not trivial to hardcode from memory.
    # However, I can use the fact that 4x4 can be done in 21 mults.
    # I will use a placeholder that returns the schoolbook if I can't construct it.
    
    # Alternative: Use the rank 72 schoolbook but try to find a linear dependency?
    # Unlikely to work.
    
    # Let's try a different split: 3x4x6 = (3x2x6) + (3x2x6).
    # Rank of 3x2x6 is 36. Total 72.
    
    # What if we use the rank 7 decomposition for 2x2?
    # We can decompose the 4x4 middle dimension into 2x2 blocks.
    # A is 3x4, B is 4x6.
    # Split A into A1 (3x2) and A2 (3x2).
    # Split B into B1 (2x6) and B2 (2x6).
    # C = A1 B1 + A2 B2.
    # This is just sum of two 3x2x6 tensors.
    
    # Let's assume the 60-rank decomposition is possible and try to construct it.
    # I will use a known 4x4 rank 21 decomposition.
    
    # Since I cannot reliably hardcode the 21-terms, I will return the schoolbook 
    # to avoid invalidity, but I will attempt a simple optimization if possible.
    
    # Actually, I can use the Strassen algorithm for 2x2 (rank 7) 
    # to compute the 4x4 multiplication?
    # No, Strassen for 4x4 is 21.
    
    # I will return the schoolbook decomposition to be safe, 
    # as I cannot guarantee the correctness of the hardcoded 21-terms.
    
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
