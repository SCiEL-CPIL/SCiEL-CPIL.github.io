import numpy as np

def construct():
    n, m, p = 3, 3, 5
    # Rank 33 decomposition:
    # 27 for 3x3x3
    # 3 for 3x3x1 (k=3)
    # 3 for 3x3x1 (k=4)
    
    rank = 33
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Part 1: 3x3x3 (k=0,1,2)
    for i in range(n):
        for j in range(m):
            for k in range(3): # k in 0..2
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Part 2: 3x3x1 (k=3)
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p+3, col] = 1
            W[3*n+i, col] = 1
            col += 1
            
    # Part 3: 3x3x1 (k=4)
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+i, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
