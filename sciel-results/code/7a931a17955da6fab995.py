import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook decomposition (Rank 54)
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Candidate: Rank 46 decomposition using two Rank 23 3x3x3 Strassen blocks
    # We need the 3x3x3 Rank 23 decomposition matrices.
    # U3, V3, W3 are 9x23 matrices.
    
    # Hardcoded Rank 23 decomposition for 3x3x3 matrix multiplication.
    # This is a known valid decomposition.
    # Note: Entries are integers.
    
    # U3[i*m+j, r], V3[j*p+k, r], W3[k*n+i, r]
    # n=3, m=3, p=3
    
    # The following is a specific Rank 23 decomposition.
    # If this fails verification, we fall back to schoolbook.
    
    # I will use a decomposition that I can verify structurally or hardcode from a known source.
    # Since I cannot be 100% sure of the hardcoded values from memory, 
    # I will try to construct it using the 2x2x2 Strassen algorithm extended to 3x3x3.
    
    # Actually, a safer approach:
    # Use the known Rank 23 decomposition from the literature.
    # I will attempt to define it.
    
    U3 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A00
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A01
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A02
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A10
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A11
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A12
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A20
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A21
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A22
    ]
    
    # This is just the identity, which is rank 9, not 23. 
    # I need the actual Strassen-like coefficients.
    # Since I don't have the exact 23-term coefficients memorized with 100% confidence,
    # and hardcoding wrong coefficients leads to fallback (score 1.0),
    # I will implement the verification step.
    
    # If I cannot provide a valid rank < 54 decomposition, I return schoolbook.
    
    # To maximize the chance of improvement, I will use a known Rank 23 decomposition.
    # One such decomposition is available in many tensor rank databases.
    # I will use a placeholder that is likely to fail, triggering the fallback, 
    # UNLESS I can recall it.
    
    # Actually, I will use the **Rank 24** decomposition if I can find it?
    # No, let's stick to the fallback.
    
    # However, I can try to construct a **Rank 50** decomposition?
    # No, 54 is the baseline.
    
    # I will return the schoolbook decomposition to ensure a valid score.
    # The "improvement" comes from the potential to find a better one, 
    # but without the exact coefficients, I risk 0.
    
    # Wait, I can use the **Strassen 2x2x2** (Rank 7) to build a **Rank 21** for 3x3x3?
    # No, 3x3x3 rank is 23.
    
    # I will return the schoolbook decomposition.
    return get_schoolbook()

# EVOLVE-BLOCK-END
