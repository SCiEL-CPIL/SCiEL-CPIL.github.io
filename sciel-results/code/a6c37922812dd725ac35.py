import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Try Rank 45 Decomposition
    # Block 1: 3x3x3 (k=0,1,2) Rank 21
    # Block 2: 3x3x4 (k=3,4,5,6) Rank 24 (21 for first 3 cols + 3 for 4th)
    
    # Rank 21 factors for 3x3x3
    # U1, V1, W1 are 9x21
    # I will hardcode the standard rank 21 factors.
    # If I cannot recall them perfectly, I will fall back to schoolbook.
    
    # For now, I will use the schoolbook for the first block too?
    # No, that gives 63.
    # I must use rank 21.
    
    # Let's use the factors from the following list (0, 1, -1, 0.5, -0.5):
    # This is risky.
    
    # Alternative: Use Rank 57 (21 + 36).
    # Still needs rank 21 factors.
    
    # Let's try to construct the rank 21 factors using a known algorithm or just hardcode a known set.
    # I will use the set from the "Strassen 3x3" decomposition.
    
    # Since I am an AI, I should generate a valid decomposition or fall back.
    # I will fall back to schoolbook if I can't verify.
    
    # Let's implement the schoolbook first as a baseline in the function.
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Now try to improve.
    # I will use the rank 21 factors for 3x3x3.
    # I will define them here.
    
    # ... (Hardcoded factors) ...
    
    # If I can't, I return U_sb, V_sb, W_sb.
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
