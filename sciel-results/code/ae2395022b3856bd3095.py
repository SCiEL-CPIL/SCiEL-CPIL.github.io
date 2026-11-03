import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # T_{2,2,2} Strassen decomposition (rank 7)
    U1 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V1 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W1 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # T_{1,2,3} schoolbook decomposition (rank 6)
    U2 = np.zeros((2, 6), dtype=float)
    V2 = np.zeros((6, 6), dtype=float)
    W2 = np.zeros((3, 6), dtype=float)
    
    col = 0
    for j in range(2):
        for k in range(3):
            U2[0*2 + j, col] = 1
            V2[j*3 + k, col] = 1
            W2[k*1 + 0, col] = 1
            col += 1
    
    # Kronecker product
    U_total = np.kron(U1, U2)
    V_total = np.kron(V1, V2)
    W_total = np.kron(W1, W2)
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
