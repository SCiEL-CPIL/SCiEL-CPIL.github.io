    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # C1
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 9 + k, c] = V4[j * 4 + k, c]
                W[k * 4 + i, c] = W4[k * 4 + i, c] # Wait, W rows are k, i.
                
    # W rows: k, i. k in 0..8, i in 0..3.
    # W[k, i, c].
    # For C1, k in 0..3.
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k * 4 + i, c] = W4[k * 4 + i, c]
                
    # C2
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 9 + (4 + k), 49 + c] = V4[j * 4 + k, c]
                
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[(4 + k) * 4 + i, 49 + c] = W4[k * 4 + i, c]
                
    # C3
    for j in range(4):
        for c in range(16):
            V[j * 9 + 8, 98 + c] = V1[j, c]
            
    for i in range(4):
        for c in range(16):
            W[8 * 4 + i, 98 + c] = W1[i, c]
