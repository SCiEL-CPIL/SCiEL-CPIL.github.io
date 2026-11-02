import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication (2, 5, 5).
    Uses the Kronecker product structure: T_{2,5,5} = T_{2,1,1} ⊗ T_{1,5,5}.
    Rank is Rank(T_{2,1,1}) * Rank(T_{1,5,5}) = 2 * 25 = 50.
    """
    n, m, p = DIMENSIONS
    
    # T_{2,1,1} decomposition (Rank 2)
    # U1: (2*1, 2), V1: (1*1, 2), W1: (1*2, 2)
    # Indices: i in 0..1, j in 0..0, k in 0..0
    # Term 0: i=0, j=0, k=0
    # Term 1: i=1, j=0, k=0
    U1 = np.array([[1, 0],
                   [0, 1]])
    V1 = np.array([[1, 1]])
    W1 = np.array([[1, 0],
                   [0, 1]])
    
    # T_{1,5,5} decomposition (Rank 25)
    # U2: (1*5, 25), V2: (5*5, 25), W2: (5*1, 25)
    # Indices: i in 0..0, j in 0..4, k in 0..4
    # Schoolbook for 1x5x5
    U2 = np.zeros((5, 25))
    V2 = np.zeros((25, 25))
    W2 = np.zeros((5, 25))
    
    col = 0
    for j in range(5):
        for k in range(5):
            # U2 index: j (since i=0 always)
            U2[j][col] = 1
            # V2 index: j*5 + k
            V2[j * 5 + k][col] = 1
            # W2 index: k (since i=0 always, W index is k*1 + 0 = k)
            W2[k][col] = 1
            col += 1
            
    # Combine using Kronecker Product
    # U = kron(U1, U2)
    # V = kron(V1, V2)
    # W = kron(W1, W2)
    
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    # Convert to lists as per interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
