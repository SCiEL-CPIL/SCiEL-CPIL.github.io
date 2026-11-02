    rank_max = 50
    U = np.zeros((n*m, rank_max))
    V = np.zeros((m*p, rank_max))
    W = np.zeros((p*n, rank_max))
    
    # ... fill ...
    
    actual_rank = col
    U = U[:, :actual_rank]
    V = V[:, :actual_rank]
    W = W[:, :actual_rank]
   