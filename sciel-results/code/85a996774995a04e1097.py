def construct():
    n, m, p = 3, 3, 6
    # Use rank 20 for 3x3x3 blocks
    rank_block = 20
    total_rank = 40
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define rank 20 decomposition for 3x3x3
    # ...
