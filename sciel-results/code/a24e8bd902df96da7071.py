import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    U_all = []
    V_all = []
    W_all = []
    
    def add_term(u, v, w):
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)
    
    # Schoolbook decomposition for 4x4 x 4x9 matrix multiplication
    # T[i,j,k] = 1 where C[i,k] = sum_j A[i,j] * B[j,k]
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(U_rows, dtype=float)
                v = np.zeros(V_rows, dtype=float)
                w = np.zeros(W_rows, dtype=float)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                add_term(u, v, w)
    
    # Convert to lists of lists
    U = [[float(x) for x in row] for row in np.array(U_all).T]
    V = [[float(x) for x in row] for row in np.array(V_all).T]
    W = [[float(x) for x in row] for row in np.array(W_all).T]
    
    return {"U": U, "V": V, "W": W}
