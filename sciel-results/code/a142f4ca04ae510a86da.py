import numpy as np

def construct():
    # Strassen 2x2 decomposition factors
    # Rows correspond to A11, A12, A21, A22
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    # Rows correspond to B11, B12, B21, B22
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    # Rows correspond to C11, C12, C21, C22
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Kronecker product to get 4x4 decomposition
    # U4 is 16x49, V4 is 16x49, W4 is 16x49
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # The problem asks for lists of lists or arrays?
    # "factor matrices of shapes (n*m, rank)..."
    # The seed program returns lists of lists.
    # I will return lists of lists to be safe, or numpy arrays if accepted.
    # The seed uses lists.
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
