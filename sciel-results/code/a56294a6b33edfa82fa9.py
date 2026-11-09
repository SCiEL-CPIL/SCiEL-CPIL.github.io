    V4_padded = np.zeros((20, 49))
    V4_padded[:16, :] = V4
    W4_padded = np.zeros((20, 49))
    W4_padded[:16, :] = W4
    
    U = np.hstack([U4, U5])
    V = np.hstack([V4_padded, V5])
    W = np.hstack([W4_padded, W5])
