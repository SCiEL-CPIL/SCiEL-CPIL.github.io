import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U, V, W = decompose(n, m, p)
    # Convert numpy arrays to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}

def decompose(n, m, p):
    # Base cases
    if n == 1 and m == 1 and p == 1:
        return np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
    
    if n == 1:
        # Rank mp
        rank = m * p
        U = np.zeros((1 * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * 1, rank))
        col = 0
        for j in range(m):
            for k in range(p):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k][col] = 1
                col += 1
        return U, V, W
        
    if m == 1:
        # Rank np
        rank = n * p
        U = np.zeros((n * 1, rank))
        V = np.zeros((1 * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return U, V, W

    if p == 1:
        # Rank nm
        rank = n * m
        U = np.zeros((n * m, rank))
        V = np.zeros((m * 1, rank))
        W = np.zeros((1 * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W

    # Special case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        return strassen_222()

    # Recursive case
    # Try splitting n
    best_rank = float('inf')
    best_decomp = None
    
    # Try splits for n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = decompose(n1, m, p)
        U2, V2, W2 = decompose(n2, m, p)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            best_decomp = (U1, V1, W1, U2, V2, W2, 'n', n1, n2)
            
    # Try splits for m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = decompose(n, m1, p)
        U2, V2, W2 = decompose(n, m2, p)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            best_decomp = (U1, V1, W1, U2, V2, W2, 'm', m1, m2)

    # Try splits for p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = decompose(n, m, p1)
        U2, V2, W2 = decompose(n, m, p2)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            best_rank = rank
            best_decomp = (U1, V1, W1, U2, V2, W2, 'p', p1, p2)

    U1, V1, W1, U2, V2, W2, axis, d1, d2 = best_decomp
    
    # Combine
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    
    if axis == 'n':
        # U: stack vertically
        U = np.vstack([U1, U2])
        # V: concat horizontally
        V = np.hstack([V1, V2])
        # W: interleave rows
        # W1 is p*n1 x R1, W2 is p*n2 x R2
        # W is p*n x (R1+R2)
        W = np.zeros((p * n, R1 + R2))
        for k in range(p):
            # Rows for k in W1: k*n1 to (k+1)*n1 - 1
            W[k*n : k*n+n1, :R1] = W1[k*n1 : (k+1)*n1, :]
            # Rows for k in W2: (k+1)*n1 to (k+1)*n1 + n2 - 1
            # In W, these are at indices k*n + n1 to k*n + n
            W[k*n+n1 : (k+1)*n, R1:] = W2[k*n2 : (k+1)*n2, :]
            
    elif axis == 'm':
        # U: interleave columns? No, U is n*m x R.
        # U1 is n*m1 x R1, U2 is n*m2 x R2
        # U is n*m x (R1+R2)
        U = np.zeros((n * m, R1 + R2))
        for i in range(n):
            # Rows for i in U1: i*m1 to (i+1)*m1 - 1
            U[i*m : i*m+m1, :R1] = U1[i*m1 : (i+1)*m1, :]
            # Rows for i in U2: (i+1)*m1 to ...
            U[i*m+m1 : (i+1)*m, R1:] = U2[i*m2 : (i+1)*m2, :]
        # V: stack vertically
        V = np.vstack([V1, V2])
        # W: concat horizontally
        W = np.hstack([W1, W2])

    else: # axis == 'p'
        # U: concat horizontally
        U = np.hstack([U1, U2])
        # V: interleave columns
        # V1 is m*p1 x R1, V2 is m*p2 x R2
        # V is m*p x (R1+R2)
        V = np.zeros((m * p, R1 + R2))
        for j in range(m):
            V[j*p : j*p+p1, :R1] = V1[j*p1 : (j+1)*p1, :]
            V[j*p+p1 : (j+1)*p, R1:] = V2[j*p2 : (j+1)*p2, :]
        # W: stack vertically
        W = np.vstack([W1, W2])

    return U, V, W

def strassen_222():
    # U is 4x7, V is 4x7, W is 4x7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W
# EVOLVE-BLOCK-END
