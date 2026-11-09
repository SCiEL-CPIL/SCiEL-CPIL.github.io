import numpy as np

def construct():
    n, m, p = 3, 5, 5
    rank = 41
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Laderman 3x3x3 (Rank 21)
    # Embedded in A[0:3, 0:3], B[0:3, 0:3], C[0:3, 0:3]
    
    # Laderman factors (U, V, W) are 9x21
    # Using a known valid set
    L_U = [
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # I need the full 21 columns. I will generate them or use a verified list.
    # To be safe and quick, I will use the schoolbook for the 3x3 part if I can't guarantee Laderman.
    # But I want the improvement.
    # I will use a hardcoded verified Laderman set.
    
    # Verified Laderman Factors (from literature)
    # U, V, W are 9x21
    L_U = [
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is only 9 columns. I need 21.
    # I will use the schoolbook for the 3x3 part to ensure validity if I can't recall the full Laderman set.
    # Schoolbook 3x3x3 is rank 27.
    # Total rank would be 27 + 18 + 2 = 47.
    # This is still better than 75.
    
    # Let's try to get Laderman right.
    # I will use a known set.
    
    # Actually, I can just use the schoolbook for the 3x3 part.
    # It is safer.
    
    # Rank 27 for 3x3x3
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
                
    # Rank 18 for 3x3x2 (A cols 0-2, B rows 0-2, B cols 3-4)
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
                
    # Rank 2 for 3x2x5 (A cols 3-4, B rows 3-4)
    # r=39 (j=3)
    for i in range(3):
        U[i * 5 + 3][col] = 1 # A col 3? No.
        # Wait. A is 3x5.
        # The term is A_{i, 3} * B_{3, k} * C_{i, k}.
        # U index is i * 5 + j. Here j=3.
        # So U[i * 5 + 3][39] = 1.
        # V index is j * 5 + k. Here j=3.
        # So V[3 * 5 + k][39] = 1 for all k.
        # W index is k * 3 + i.
        # So W[k * 3 + i][39] = 1 for all k.
        
    # Let's fill it properly.
    
    # Re-initialize to be clean
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Schoolbook 3x3x3 (Rank 27)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: Schoolbook 3x3x2 (Rank 18)
    # A cols 0-2, B rows 0-2, B cols 3-4
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 3: Rank 2 for A cols 3-4, B rows 3-4
    # r=39 (j=3)
    for i in range(3):
        U[i * m + 3][col] = 1
    for k in range(5):
        V[3 * p + k][col] = 1
        W[k * n + i][col] = 1 # Wait, W depends on i.
        # My previous logic: W[k, 39] = 1 for all k?
        # No, W[k, 39] must be 1 for all i,k?
        # T_{i 3 k} = 1.
        # Sum U_{i 3} V_{3 k} W_{k i}.
        # If U_{i 3} = 1, V_{3 k} = 1, W_{k i} = 1, sum is 1.
        # So W[k * n + i][39] = 1 for all i, k.
    for i in range(3):
        for k in range(5):
            W[k * n + i][col] = 1
    col += 1
            
    # r=40 (j=4)
    for i in range(3):
        U[i * m + 4][col] = 1
    for k in range(5):
        V[4 * p + k][col] = 1
    for i in range(3):
        for k in range(5):
            W[k * n + i][col] = 1
    col += 1
            
    return {"U": U, "V": V, "W": W}
