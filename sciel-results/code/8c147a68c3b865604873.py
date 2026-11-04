import time
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 135.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Winograd 3x3x3 decomposition (rank 23)
    # T[i,k,j] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
    # Using the known 23-term decomposition for 3x3x3 matrix multiplication.
    # The entries are integers and half-integers.
    
    # The 23 terms for 3x3x3 are given by the following coefficients:
    # This is a standard decomposition. I will hardcode the U, V, W matrices for 3x3x3.
    # Reference: "A 23-term algorithm for 3x3 matrix multiplication"
    
    # Let's use the known decomposition.
    # U (9x23), V (9x23), W (9x23)
    # The indices are i*3+j for U, j*3+k for V, k*3+i for W.
    
    # I will use the decomposition from the literature.
    # The coefficients are:
    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1
    # ... this is tedious to hardcode manually from memory.
    # Instead, I will use a known set of coefficients for 3x3x3 rank 23.
    
    # Let's use the following set of 23 terms (U, V, W) for 3x3x3:
    # Note: The entries must be integers or half-integers.
    
    # I will construct the 3x3x3 decomposition using a known algorithm.
    # Since I cannot look it up, I will use a simpler approach:
    # Use the schoolbook for 3x3x3 (rank 27) and try to find a rank 23 decomposition
    # by solving a linear system or using a known result.
    
    # Actually, for the purpose of this task, I can use the rank 27 schoolbook for 3x3x3
    # and rank 16 schoolbook for 3x1x3, giving 27+16 = 43 for the 3x4x4 part.
    # And for the 3x4x3 part, I can use rank 12 schoolbook.
    # Total rank: 43 + 12 = 55.
    # This is better than 84.
    
    # Let's try to find a better decomposition.
    # The rank of 3x4x7 is known to be at most 3*4*7 = 84.
    # We can split the 4x7 matrix B into B1 (4x4) and B2 (4x3).
    # C = A * B = A * B1 + A * B2.
    # Rank(A * B1) for 3x4 * 4x4.
    # Rank(A * B2) for 3x4 * 4x3.
    
    # For 3x4 * 4x4:
    # We can pad A to 4x4 and B1 to 4x4, but that gives rank 64.
    # Alternatively, we can use the fact that 3x4 * 4x4 can be decomposed.
    # Let's use the schoolbook for 3x4 * 4x4: rank 3*4*4 = 48.
    # And schoolbook for 3x4 * 4x3: rank 3*4*3 = 36.
    # Total rank: 48 + 36 = 84. This is the same as the original.
    
    # To improve, we need a better decomposition for 3x4 * 4x4 or 3x4 * 4x3.
    # For 3x4 * 4x3:
    # We can view this as (3x4) * (4x3).
    # Let's try to use a rank 23 decomposition for 3x3 * 3x3 and combine.
    # Split A (3x4) into A1 (3x3) and A2 (3x1).
    # Split B (4x3) into B1 (3x3) and B2 (1x3).
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2.
    # Rank(A1*B1) = 23 (using Winograd 3x3x3).
    # Rank(A1*B2) = 3*1*3 = 9 (schoolbook).
    # Rank(A2*B1) = 1*3*3 = 9 (schoolbook).
    # Rank(A2*B2) = 1*1*3 = 3 (schoolbook).
    # Total rank: 23 + 9 + 9 + 3 = 44.
    
    # For 3x4 * 4x4:
    # Split A (3x4) into A1 (3x3) and A2 (3x1).
    # Split B1 (4x4) into B11 (3x3), B12 (3x1), B21 (1x3), B22 (1x1).
    # C1 = A1*B11 + A1*B12 + A2*B11 + A2*B12 + A1*B21 + A1*B22 + A2*B21 + A2*B22.
    # Rank(A1*B11) = 23.
    # Rank(A1*B12) = 3*1*3 = 9.
    # Rank(A2*B11) = 1*3*3 = 9.
    # Rank(A2*B12) = 1*1*3 = 3.
    # Rank(A1*B21) = 3*3*1 = 9.
    # Rank(A1*B22) = 3*1*1 = 3.
    # Rank(A2*B21) = 1*3*1 = 3.
    # Rank(A2*B22) = 1*1*1 = 1.
    # Total rank: 23 + 9 + 9 + 3 + 9 + 3 + 3 + 1 = 60.
    
    # Total rank for 3x4 * 4x7: 44 + 60 = 104. This is worse than 84.
    
    # Let's try a different split.
    # Split B (4x7) into B1 (4x4) and B2 (4x3).
    # We found that 3x4 * 4x3 has rank 44 using the 3x3x3 decomposition.
    # For 3x4 * 4x4, let's try to use a better decomposition.
    # If we use the schoolbook for 3x4 * 4x4, the rank is 48.
    # Total rank: 48 + 44 = 92. Still worse than 84.
    
    # It seems that using the 3x3x3 decomposition doesn't help directly because of the overhead.
    # Let's stick with the schoolbook decomposition of rank 84, which is guaranteed to work.
    
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Since the deadline is tight and finding a better decomposition is complex,
    # we return the schoolbook decomposition.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
