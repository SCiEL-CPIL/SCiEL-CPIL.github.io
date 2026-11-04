import numpy as np

def construct():
    # Strassen 2x2x2
    U2, V2, W2 = strassen_2x2_decomp()
    
    # Strassen 4x4x4 (Tensor product of two 2x2x2)
    U4, V4, W4 = tensor_product(U2, V2, W2, U2, V2, W2)
    
    # M_4,4,1
    U441 = np.zeros((16, 4))
    V441 = np.zeros((20, 4))
    W441 = np.zeros((20, 4))
    
    for j in range(4):
        for i in range(4):
            U441[i * 4 + j, j] = 1
            V441[j * 5 + 4, j] = 1
            W441[4 * 4 + i, j] = 1
            
    # Combine
    U = np.hstack([U4, U441])
    V = np.hstack([V4, V441])
    W = np.hstack([W4, W441])
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def strassen_2x2_decomp():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def tensor_product(U1, V1, W1, U2, V2, W2):
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    n1 = int(np.sqrt(U1.shape[0]))
    n2 = int(np.sqrt(U2.shape[0]))
    
    U = np.zeros((n1*n1*r, r))
    V = np.zeros((n1*n1*r, r))
    W = np.zeros((n1*n1*r, r))
    
    for i in range(r1):
        for j in range(r2):
            c = i * r2 + j
            U[:, c] = np.kron(U1[:, i], U2[:, j])
            V[:, c] = np.kron(V1[:, i], V2[:, j])
            W[:, c] = np.kron(W1[:, i], W2[:, j])
            
    return U, V, W
