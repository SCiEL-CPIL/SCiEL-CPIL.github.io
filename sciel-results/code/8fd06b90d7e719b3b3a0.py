def generate_rank_56(n, m, p):
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    def add_strassen_block(j0, k0, base_col):
        # ... same as before ...
        return base_col + 7

    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            col = add_strassen_block(j0, k0, col)
            
    return U, V, W
