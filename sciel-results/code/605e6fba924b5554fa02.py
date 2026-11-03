import numpy as np

def construct():
    # Strassen M_2,2,2
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [1, 1, 0, 1, 0, 1, 0],  # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1
    ])
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0], # j=0, w=0
        [1, 0, 1, 0, 0, 1, 1],  # j=0, w=1
        [1, 0, -1, 1, 0, 0, 1], # j=1, w=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1, w=1
    ])
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, 1], # i=0, w=0
        [0, 0, 1, 0, 1, 0, 0],  # i=0, w=1
        [0, 1, 0, 1, 0, 0, 0],  # i=1, w=0
        [1, -1, 1, 0, 0, 1, 0]  # i=1, w=1
    ])
    
    # Standard M_1,2,2
    U_V = np.array([
        [1, 1, 0, 0], # u=0
        [0, 0, 1, 1]  # u=1
    ])
    V_V = np.array([
        [1, 0, 0, 0], # u=0, t=0
        [0, 1, 0, 0], # u=0, t=1
        [0, 0, 1, 0], # u=1, t=0
        [0, 0, 0, 1]  # u=1, t=1
    ])
    W_V = np.array([
        [1, 0, 1, 0], # t=0
        [0, 1, 0, 1]  # t=1
    ])
    
    # Tensor Product
    # U_new: 8x28
    U_new = np.zeros((8, 28))
    for i in range(2):
        for j in range(2):
            for u in range(2):
                r = i * 4 + j * 2 + u
                for cS in range(7):
                    for cV in range(4):
                        c = cS * 4 + cV
                        U_new[r, c] = U_S[i * 2 + j, cS] * U_V[u, cV]
                        
    # V_new: 16x28
    V_new = np.zeros((16, 28))
    for j in range(2):
        for u in range(2):
            for w in range(2):
                for t in range(2):
                    r = j * 8 + u * 4 + w * 2 + t
                    for cS in range(7):
                        for cV in range(4):
                            c = cS * 4 + cV
                            V_new[r, c] = V_S[j * 2 + w, cS] * V_V[u * 2 + t, cV]
                            
    # W_new: 8x28
    W_new = np.zeros((8, 28))
    for i in range(2):
        for w in range(2):
            for t in range(2):
                r = i * 4 + w * 2 + t
                for cS in range(7):
                    for cV in range(4):
                        c = cS * 4 + cV
                        W_new[r, c] = W_S[i * 2 + w, cS] * W_V[t, cV]
                        
    return {"U": U_new, "V": V_new, "W": W_new}
