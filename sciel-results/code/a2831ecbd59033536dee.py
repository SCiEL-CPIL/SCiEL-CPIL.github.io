    # Combine
    # U: (n*m, rank1+rank2)
    # U1 is (n*m1, rank1). U2 is (n*m2, rank2).
    # We need to place them in the correct rows and columns.
    rank_total = U1.shape[1] + U2.shape[1]
    U_combined = np.zeros((n * m, rank_total))
    U_combined[:n * m1, :U1.shape[1]] = U1
    U_combined[n * m1:, U1.shape[1]:] = U2
    
    # V: (m*p, rank1+rank2)
    # V1 is (m1*p, rank1). V2 is (m2*p, rank2).
    V_combined = np.zeros((m * p, rank_total))
    V_combined[:m1 * p, :V1.shape[1]] = V1
    V_combined[m1 * p:, V1.shape[1]:] = V2
    
    # W: (p*n, rank1+rank2)
    # W1 and W2 have same shape (p*n, rank).
    W_combined = np.hstack([W1, W2])
