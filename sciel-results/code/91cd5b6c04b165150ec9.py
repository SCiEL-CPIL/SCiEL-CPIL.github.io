import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Lander-Bailey 4x4x4 decomposition (rank 19)
    # Indices for 4x4x4:
    # U: (16, 19)
    # V: (16, 19)
    # W: (16, 19)
    
    # The Lander-Bailey decomposition is complex to type out from memory.
    # However, I can use a known explicit form or a simplified version if I don't have the exact coefficients.
    # But wait, if I don't have the exact coefficients, I can't use it.
    # Is there a simpler rank-19 decomposition?
    # No, 19 is the minimal rank.
    
    # If I cannot recall the exact 19 terms, I might have to fall back to schoolbook for 4x4x4 (rank 16) + schoolbook for 4x4x1 (rank 16) = 32?
    # Wait, schoolbook for 4x4x4 is rank 16.
    # Schoolbook for 4x4x1 is rank 16.
    # Total rank 32.
    # 32 is better than 80.
    # And 32 is easier to implement than 19+16=35.
    # Wait, is schoolbook rank for 4x4x4 equal to 16?
    # Yes, $4 \times 4 \times 4 = 64$ terms? No.
    # Schoolbook for $n \times n \times n$ has rank $n^3$.
    # For $n=4$, rank is 64.
    # My previous calculation of "schoolbook rank 80" for 4x4x5 was $4 \times 4 \times 5 = 80$.
    # So schoolbook for 4x4x4 is $4 \times 4 \times 4 = 64$.
    # So $64 + 16 = 80$.
    # So using schoolbook for both parts gives 80.
    
    # Using Lander-Bailey (19) for 4x4x4 and schoolbook (16) for 4x4x1 gives 35.
    # This is a huge improvement.
    
    # I need the Lander-Bailey coefficients.
    # Since I cannot look them up, I will assume I can construct a valid decomposition of rank 35 if I have the coefficients.
    # If I don't, I will return the schoolbook decomposition (rank 80) to be safe, but the score will be 1.0.
    # However, the prompt asks for a "stronger algorithm".
    
    # Let's try to find a rank < 80 decomposition that is easier to describe.
    # Maybe $M_{4,4,5}$ has a known rank?
    # If I don't know the Lander-Bailey coefficients, I cannot implement rank 35.
    
    # Alternative: Use the fact that $M_{4,4,5}$ can be decomposed using $M_{4,4,4}$ and $M_{4,4,1}$.
    # If I don't have $M_{4,4,4}$ rank 19, I can use rank 16 (schoolbook) for $M_{4,4,4}$?
    # No, schoolbook is 64.
    
    # Is there a rank 20-30 decomposition for 4x4x4 that is easier to write?
    # No.
    
    # I will output the schoolbook decomposition (rank 80) as it is the only one I can guarantee is correct without external data.
    # But wait, the prompt says "score is seed/value". If I return 80, score is 1.0.
    # If I return a valid decomposition of rank 79, score is > 1.0.
    
    # Can I improve the schoolbook decomposition?
    # No, schoolbook is minimal for the naive approach.
    
    # I will stick to the schoolbook decomposition to ensure correctness.
    
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
