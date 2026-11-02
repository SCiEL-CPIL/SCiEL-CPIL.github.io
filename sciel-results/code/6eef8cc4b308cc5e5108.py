import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n * m * p
    rank = n * m * p
    
    U = []
    V = []
    W = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                # U[i*m + j] = 1
                u[i * m + j] = 1
                # V[j*p + k] = 1
                v[j * p + k] = 1
                # W[k*n + i] = 1
                w[k * n + i] = 1
                
                U.append(u)
                V.append(v)
                W.append(w)
                
    # Transpose to required shapes
    U_out = np.array(U).T.tolist()
    V_out = np.array(V).T.tolist()
    W_out = np.array(W).T.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
