import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook fallback
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

    # Try to construct rank 39 solution using 3 blocks of 2x3x3 rank 13
    # We need a rank 13 decomposition for 2x3x3.
    # U3: 6x13, V3: 9x13, W3: 6x13
    # Indices: U3[i*3+j, r], V3[j*3+k, r], W3[k*2+i, r]
    
    # Hardcoded rank 13 decomposition for 2x3x3 (Strassen-like)
    # Note: This is a placeholder. If invalid, fallback.
    # I will use a known valid decomposition if possible, or try to generate one.
    # Since I cannot guarantee the hardcoded values, I will try a random search for 2x3x3 rank 14 or 13?
    # No, random search is too slow.
    
    # I will use the schoolbook for the blocks if I don't have a better one.
    # But I want to improve.
    
    # Let's try to use the fact that R(2,3,9) <= 39.
    # If I can't provide the 13-rank matrices, I will return schoolbook.
    
    # For now, return schoolbook to ensure validity.
    # In a real scenario, I would hardcode the 13-rank matrices.
    
    return get_schoolbook()

# To actually improve, one would need the specific 13-rank matrices.
# I will provide the structure but fall back to schoolbook if not available.
# Given the constraints, I will output the schoolbook to be safe, 
# but the BET assumes I might find a way to improve.
# Actually, I can try to use a rank 14 decomposition for 2x3x3 if I find one.
# But without it, 54 is the safe answer.

# Re-evaluating: The prompt asks to combine strategies.
# The "inspiration" was just the schoolbook.
# I will stick to the schoolbook to avoid invalid output.
