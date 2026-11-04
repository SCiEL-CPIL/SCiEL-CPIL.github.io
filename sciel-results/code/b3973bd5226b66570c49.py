<<<<<<< SEARCH
    # Left block (columns 0-3 of B, i.e., k=0..3 in C)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]
            
    # Right block (columns 4-7 of B, i.e., k=4..7 in C)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]
=======
    # Left block (columns 0-3 of B, i.e., k=0..3 in C)
    U[:, 0:49] = U4
    # V: B[j,k] for k=0..3 -> index j*8+k; V4[j,k] for k=0..3 -> index j*4+k
    for j in range(4):
        V[j * 8 + 0:4, 0:49] = V4[j * 4 + 0:4, :]
        W[0 * 4 + j:4 * 4 + j:4, 0:49] = W4[0 * 4 + j:4 * 4 + j:4, :]
    # More explicit for W: W[k*4+i] for k=0..3, i=j
    for k in range(4):
        W[k * 4 + 0:4, 0:49] = W4[k * 4 + 0:4, :]
    # Actually let's be very explicit
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]
            
    # Right block (columns 4-7 of B, i.e., k=4..7 in C)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]
>>>>>>> REPLACE

Let me reconsider the verification - it's actually quite expensive for rank 98. Let me simplify it to just a spot check or remove it to save time, and focus on making sure the Kronecker construction is correct.

