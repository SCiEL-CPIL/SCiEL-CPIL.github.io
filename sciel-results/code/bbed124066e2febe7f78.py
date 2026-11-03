import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Attempt Rank 32 Decomposition
    # Part 1: 3x3x3 (first 3 columns of B)
    # Part 2: 3x3x1 (4th column of B)
    
    # Hardcoded Rank 23 decomposition for 3x3x3
    # u, v, w are 3-vectors
    # Source: Bini, Lotti, Romani (1979)
    # Note: I need to be sure about these coefficients.
    # If I am not sure, I should use schoolbook.
    # Let's try to use a known valid set.
    
    # Actually, I will use a simpler strategy if I can't recall the 23 terms.
    # But 32 is a significant improvement.
    
    # Let's use a placeholder for the 23 terms and verify.
    # If verification fails, return schoolbook.
    
    terms_333 = []
    # ... (I will insert the 23 terms here if I can find them)
    
    # Since I cannot guarantee the 23 terms from memory, I will use the schoolbook for 3x3x3 (rank 27)
    # and schoolbook for 3x3x1 (rank 9). Total 36.
    # This is not an improvement.
    
    # Is there a simpler improvement?
    # What if I use Strassen for 2x2x2 blocks?
    # 3x3x3 can be split into:
    # 2x2x2 (rank 7)
    # 2x1x3 (rank 6? 2*1*3=6)
    # 1x2x3 (rank 6? 1*2*3=6)
    # 1x1x3 (rank 3? 1*1*3=3)
    # Total 7+6+6+3 = 22?
    # But the blocks are not disjoint in the standard way.
    
    # Let's stick to the safe schoolbook 36 if I can't find a better one.
    # But the hint says 34 was achieved.
    # Maybe I can achieve 34 by a simple modification?
    
    # Let's return schoolbook for now to ensure validity.
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
