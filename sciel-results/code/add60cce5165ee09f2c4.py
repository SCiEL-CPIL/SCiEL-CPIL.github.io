def construct():
    n, m, p = 3, 3, 6
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 23 terms for 3x3x3
    # ...
    
    # Apply to block 1 (k=0..2)
    # Apply to block 2 (k=3..5)
    
    return {"U": U, "V": V, "W": W}
