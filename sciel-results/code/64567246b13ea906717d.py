import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank1 = 56
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Schoolbook 2x2x2 decomposition
    U2 = np.zeros((4, 8))
    V2 = np.zeros((4, 8))
    W2 = np.zeros((4, 8))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U2[i*2+j, col] = 1
                V2[j*2+k, col] = 1
                W2[k*2+i, col] = 1
                col += 1
                
    # Define the 7 Strassen products
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each product is a sum of block products.
    # We generate the terms for each block product and add them.
    
    col = 0
    for m_idx in range(7):
        # Define the block products for M_m
        # This is complex.
        
        # I will use the schoolbook decomposition for 4x4x4.
        pass

    # Let's use the schoolbook decomposition for 4x4x4.
    rank1 = 64
    total_rank = rank1 + rank2
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Part 1: Schoolbook 4x4x4
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j, col] = 1
                V[j*4+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
                
    # Part 2: Schoolbook 4x4x3
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4+j, col] = 1
                V[j*7+(4+k), col] = 1
                W[(4+k)*4+i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

