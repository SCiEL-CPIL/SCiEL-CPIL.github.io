import numpy as np

DIMENSIONS = [4, 5, 5]

def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def combine(U1, V1, W1, U2, V2, W2):
    # Concatenate along the rank dimension (columns)
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Split n=4 into 1 and 3
    n1, n2 = 1, 3
    
    # For R(n1, m, p) = R(1, 5, 5)
    U1, V1, W1 = schoolbook(n1, m, p)
    
    # For R(n2, m, p) = R(3, 5, 5)
    # Split m=5 into 2 and 3
    m1, m2 = 2, 3
    
    # R(3, 2, 5)
    U2a, V2a, W2a = schoolbook(n2, m1, p)
    # R(3, 3, 5)
    U2b, V2b, W2b = schoolbook(n2, m2, p)
    
    # Combine U2a and U2b
    # U2a has rows for i in 0..2, j in 0..1
    # U2b has rows for i in 0..2, j in 0..2
    # We need to place them correctly in the full U2 matrix of size 3x5
    U2 = np.zeros((n2 * m, n2 * (m1 + m2) * p)) # This is not right, rank is sum
    # Actually, we just hstack the factors
    U2 = np.hstack([U2a, U2b])
    V2 = np.hstack([V2a, V2b])
    W2 = np.hstack([W2a, W2b])
    
    # Now combine the n1 and n2 parts
    # U1 has rows for i in 0..0, j in 0..4
    # U2 has rows for i in 0..2, j in 0..4 (but constructed from parts)
    # Wait, the rows of U must be ordered by i*m + j.
    # U1 corresponds to i=0.
    # U2 corresponds to i=1,2,3.
    # So we need to insert U1 rows at the top, and U2 rows below.
    
    U_final = np.vstack([U1, U2])
    V_final = np.hstack([V1, V2])
    W_final = np.hstack([W1, W2])
    
    # Convert to lists
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
