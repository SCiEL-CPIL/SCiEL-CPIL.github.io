import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # 7 blocks of 2x2x2 (Strassen, rank 7 each) + 1 block of 2x2x1 (rank 4)
    total_rank = 7 * 7 + 4  # = 53
    
    U = np.zeros((n * m, total_rank))  # (4, 53)
    V = np.zeros((m * p, total_rank))  # (30, 53)
    W = np.zeros((p * n, total_rank))  # (30, 53)
    
    col = 0
    
    # 7 Strassen blocks for 2x2x2 subproblems
    for block in range(7):
        ko = block * 2  # k offset (starting column in B, starting row in C)
        
        for prod in range(7):
            c = col + prod
            
            # U entries (A row index = i*m + j)
            if prod == 0:   # A00+A11
                U[0, c] = 1; U[3, c] = 1
            elif prod == 1: # A10+A11
                U[2, c] = 1; U[3, c] = 1
            elif prod == 2: # A00
                U[0, c] = 1
            elif prod == 3: # A11
                U[3, c] = 1
            elif prod == 4: # A00+A01
                U[0, c] = 1; U[1, c] = 1
            elif prod == 5: # A10-A00
                U[2, c] = 1; U[0, c] = -1
            elif prod == 6: # A01-A11
                U[1, c] = 1; U[3, c] = -1
            
            # V entries (B row index = j*p + k)
            if prod == 0:   # B[0,ko] + B[1,ko+1]
                V[0*p + ko, c] = 1; V[1*p + (ko+1), c] = 1
            elif prod == 1: # B[0,ko+1]
                V[0*p + (ko+1), c] = 1
            elif prod == 2: # B[1,ko] - B[1,ko+1]
                V[1*p + ko, c] = 1; V[1*p + (ko+1), c] = -1
            elif prod == 3: # B[0,ko+1] - B[1,ko]
                V[0*p + (ko+1), c] = 1; V[1*p + ko, c] = -1
            elif prod == 4: # B[1,ko+1]
                V[1*p + (ko+1), c] = 1
            elif prod == 5: # B[0,ko] + B[1,ko]
                V[0*p + ko, c] = 1; V[1*p + ko, c] = 1
            elif prod == 6: # B[0,ko+1] + B[1,ko+1]
                V[0*p + (ko+1), c] = 1; V[1*p + (ko+1), c] = 1
            
            # W entries (C row index = k*n + i)
            if prod == 0:   # C[ko,0] += 1, C[ko+1,1] += 1
                W[ko*n + 0, c] = 1; W[(ko+1)*n + 1, c] = 1
            elif prod == 1: # C[ko,1] += 1, C[ko+1,0] += 1
                W[ko*n + 1, c] = 1; W[(ko+1)*n + 0, c] = 1
            elif prod == 2: # C[ko,1] += 1, C[ko+1,1] += -1
                W[ko*n + 1, c] = 1; W[(ko+1)*n + 1, c] = -1
            elif prod == 3: # C[ko,0] += 1, C[ko+1,0] += 1
                W[ko*n + 0, c] = 1; W[(ko+1)*n + 0, c] = 1
            elif prod == 4: # C[ko,0] += -1, C[ko+1,1] += 1
                W[ko*n + 0, c] = -1; W[(ko+1)*n + 1, c] = 1
            elif prod == 5: # C[ko+1,1] += 1
                W[(ko+1)*n + 1, c] = 1
            elif prod == 6: # C[ko,0] += 1
                W[ko*n + 0, c] = 1
        
        col += 7
    
    # 1 schoolbook block for the remaining 2x2x1
    ko = 14
    for i in range(n):
        for j in range(m):
            c = col
            U[i * m + j, c] = 1
            V[j * p + ko, c] = 1
            W[ko * n + i, c] = 1
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
