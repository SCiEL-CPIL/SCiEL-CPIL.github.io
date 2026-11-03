def _schoolbook_3x3x3():
    """Returns U, V, W for 3x3x3 matmul with rank 9."""
    n, m, p = 3, 3, 3
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _schoolbook_3x3x1():
    """Returns U, V, W for 3x3x1 matmul with rank 3."""
    n, m, p = 3, 3, 1
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    # Full problem: n=3, m=3, p=4
    n, m, p = 3, 3, 4
    total_rank = 12
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: 3x3x3 (columns 0-2 of B)
    U1, V1, W1 = _schoolbook_3x3x3()
    # U1 is 9x9, V1 is 9x9, W1 is 9x9
    U[:, 0:9] = U1
    # V1 corresponds to B indices j*3+k for k in 0..2.
    # In full V (12x12), rows 0..8 correspond to k=0..2.
    V[0:9, 0:9] = V1
    # W1 corresponds to C indices k*3+i for k in 0..2.
    # In full W (12x12), rows 0..8 correspond to k=0..2.
    W[0:9, 0:9] = W1
    
    # Part 2: 3x3x1 (column 3 of B)
    U2, V2, W2 = _schoolbook_3x3x1()
    # U2 is 9x3, V2 is 3x3, W2 is 3x3
    U[:, 9:12] = U2
    # V2 corresponds to B indices j*1+0 for j in 0..2.
    # In full V (12x12), rows 9..11 correspond to k=3 (since p=4, index is j*4+3? No.)
    # Wait, V index is j*p + k.
    # For p=4, k=3. Indices are j*4 + 3.
    # j=0 -> 3. j=1 -> 7. j=2 -> 11.
    # So rows 3, 7, 11 in V.
    # My V2 is 3x3. Row j in V2 corresponds to B[j, 0] in the 3x3x1 problem.
    # This maps to B[j, 3] in the full problem.
    # So V2[0, :] -> V[3, 9:12]
    # V2[1, :] -> V[7, 9:12]
    # V2[2, :] -> V[11, 9:12]
    V[3, 9:12] = V2[0, :]
    V[7, 9:12] = V2[1, :]
    V[11, 9:12] = V2[2, :]
    
    # W2 corresponds to C indices k*3+i for k=0.
    # In full W (12x12), index is k*3 + i? No, W index is k*n + i.
    # For p=4, n=3. k ranges 0..3.
    # W index is k*3 + i.
    # For k=3, indices are 9, 10, 11.
    # So W2[0, :] -> W[9, 9:12]
    # W2[1, :] -> W[10, 9:12]
    # W2[2, :] -> W[11, 9:12]
    W[9, 9:12] = W2[0, :]
    W[10, 9:12] = W2[1, :]
    W[11, 9:12] = W2[2, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
