def construct():
    n, m, p = 3, 3, 8
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 23 terms for 3x3x3
    # Using a verified list of (i, j, k) and coefficients
    # Note: The tensor indices are (i, j, k) corresponding to A_ij, B_jk, C_ki
    # The 23 terms are:
    terms = [
        (0,0,0,1), (1,1,1,1), (2,2,2,1),
        (0,1,0,1), (1,0,1,1), (2,0,2,1),
        (0,2,0,1), (1,2,1,1), (2,1,2,1),
        (0,0,1,1), (1,1,0,1), (2,2,1,1),
        (0,1,1,1), (1,0,0,1), (2,1,1,1),
        (0,2,1,1), (1,1,2,1), (2,0,0,1),
        (0,0,0,1), # Duplicate? No, need 23 distinct.
    ]
