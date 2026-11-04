import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Strassen 2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, -1, 0, 0, 1, 0]
    ])
    
    # Schoolbook 3x3
    n3 = 3
    rank3 = n3**3
    U3 = np.zeros((n3*n3, rank3))
    V3 = np.zeros((n3*n3, rank3))
    W3 = np.zeros((n3*n3, rank3))
    c = 0
    for i in range(n3):
        for j in range(n3):
            for k in range(n3):
                U3[i*n3 + j, c] = 1
                V3[j*n3 + k, c] = 1
                W3[k*n3 + i, c] = 1
                c += 1
                
    # Kronecker product
    U6_kron = np.kron(U3, U2)
    V6_kron = np.kron(V3, V2)
    W6_kron = np.kron(W3, W2)
    
    # Reorder rows
    U6 = np.zeros_like(U6_kron)
    V6 = np.zeros_like(V6_kron)
    W6 = np.zeros_like(W6_kron)
    
    # U6: i, j
    for i in range(6):
        for j in range(6):
            i3, i2 = divmod(i, 2)
            j3, j2 = divmod(j, 2)
            kron_idx = i3 * 12 + j3 * 4 + i2 * 2 + j2
            req_idx = i * 6 + j
            U6[req_idx] = U6_kron[kron_idx]
            
    # V6: j, k
    for j in range(6):
        for k in range(6):
            j3, j2 = divmod(j, 2)
            k3, k2 = divmod(k, 2)
            kron_idx = j3 * 12 + k3 * 4 + j2 * 2 + k2
            req_idx = j * 6 + k
            V6[req_idx] = V6_kron[kron_idx]
            
    # W6: k, i
    for k in range(6):
        for i in range(6):
            k3, k2 = divmod(k, 2)
            i3, i2 = divmod(i, 2)
            kron_idx = k3 * 12 + i3 * 4 + k2 * 2 + i2
            req_idx = k * 6 + i
            W6[req_idx] = W6_kron[kron_idx]
            
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
