import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose 5 into 3 + 2
    # A is 3x5 = [A1 (3x3) | A2 (3x2)]
    # B is 5x5 = [[B11 (3x3), B12 (3x2)], [B21 (2x3), B22 (2x2)]]
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    
    # We will construct U, V, W for each block and concatenate them.
    # Total rank is sum of ranks of blocks.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Block 1: A1 (3x3) * B11 (3x3) -> C11 (3x3)
    # Rank 27
    U1, V1, W1 = _decomp_3x3x3()
    U_blocks.append(U1)
    V_blocks.append(V1)
    W_blocks.append(W1)
    
    # Block 2: A1 (3x3) * B12 (3x2) -> C12 (3x2)
    # Rank 6
    U2, V2, W2 = _decomp_3x3x2()
    U_blocks.append(U2)
    V_blocks.append(V2)
    W_blocks.append(W2)
    
    # Block 3: A2 (3x2) * B21 (2x3) -> C21 (3x3)
    # Rank 6
    U3, V3, W3 = _decomp_3x2x3()
    U_blocks.append(U3)
    V_blocks.append(V3)
    W_blocks.append(W3)
    
    # Block 4: A2 (3x2) * B22 (2x2) -> C22 (3x2)
    # Rank 6
    U4, V4, W4 = _decomp_3x2x2()
    U_blocks.append(U4)
    V_blocks.append(V4)
    W_blocks.append(W4)
    
    # Concatenate
    U = np.concatenate(U_blocks, axis=1)
    V = np.concatenate(V_blocks, axis=1)
    W = np.concatenate(W_blocks, axis=1)
    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _decomp_3x3x3():
    n, m, p = 3, 3, 3
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

def _decomp_3x3x2():
    n, m, p = 3, 3, 2
    rank = 6
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    # k=1
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 0, col] = 1
            W[0 * n + i, col] = 1
            col += 1
    # k=2
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 1, col] = 1
            W[1 * n + i, col] = 1
            col += 1
    return U, V, W

def _decomp_3x2x3():
    n, m, p = 3, 2, 3
    rank = 6
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    # j=1
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    # j=2
    for i in range(n):
        for k in range(p):
            U[i * m + 1, col] = 1
            V[1 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    return U, V, W

def _decomp_3x2x2():
    n, m, p = 3, 2, 2
    rank = 6
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    # j=1
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    # j=2
    for i in range(n):
        for k in range(p):
            U[i * m + 1, col] = 1
            V[1 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    return U, V, W
# EVOLVE-BLOCK-END
