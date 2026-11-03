<<<<<<< SEARCH
    # Fill first half (k < 4)
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i*4 + j, R] = U4[i, j, R]
                
    for j in range(4):
        for k in range(4):
            for R in range(49):
                V[j*8 + k, R] = V4[j, k, R]
                
    for k in range(4):
        for i in range(4):
            for R in range(49):
                W[k*4 + i, R] = W4[k, i, R]
                
    # Fill second half (k >= 4)
    # Map k in 4..7 to k' in 0..3
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i*4 + j, 49 + R] = U4[i, j, R]
                
    for j in range(4):
        for kp in range(4):
            k = 4 + kp
            for R in range(49):
                V[j*8 + k, 49 + R] = V4[j, kp, R]
                
    for kp in range(4):
        k = 4 + kp
        for i in range(4):
            for R in range(49):
                W[k*4 + i, 49 + R] = W4[kp, i, R]
=======
    # Fill first half (k < 4)
    U[:, :49] = U4.reshape(16, 49)
    V[:, :49] = V4[:, :4, :].reshape(32, 49)
    W[:, :49] = W4[:, :4, :].reshape(32, 49)
    
    # Fill second half (k >= 4)
    U[:, 49:] = U4.reshape(16, 49)
    V[:, 49:] = V4[:, 4:, :].reshape(32, 49)
    W[:, 49:] = W4[:, 4:, :].reshape(32, 49)
>>>>>>> REPLACE
