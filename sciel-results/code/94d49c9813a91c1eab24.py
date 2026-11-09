def construct():
    n, m, p = DIMENSIONS
    # Rank 74
    rank = 74
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 1. Strassen Block for i in {0,1}, j in {0,1}, k in {0,1}
    # Indices in U: i*m + j
    # Indices in V: j*p + k
    # Indices in W: k*n + i
    
    # P1
    U[0*5+0, 0] += 1; U[1*5+1, 0] += 1
    V[0*5+0, 0] += 1; V[1*5+1, 0] += 1
    W[0*3+0, 0] += 1; W[1*3+1, 0] += 1 # C00, C11
    
    # P2
    U[1*5+0, 1] += 1; U[1*5+1, 1] += 1
    V[0*5+0, 1] += 1
    W[0*3+1, 1] += 1 # C10
    
    # P3
    U[0*5+0, 2] += 1
    V[1*5+0, 2] += 1; V[0*5+0, 2] -= 1
    W[0*3+0, 2] -= 1; W[1*3+1, 2] -= 1 # C00, C11
    
    # P4
    U[1*5+1, 3] += 1
    V[0*5+1, 3] += 1; V[0*5+0, 3] -= 1
    W[0*3+0, 3] -= 1; W[0*3+1, 3] += 1 # C00, C10
    
    # P5
    U[0*5+0, 4] += 1; U[0*5+1, 4] += 1
    V[1*5+1, 4] += 1
    W[0*3+0, 4] -= 1; W[1*3+0, 4] += 1 # C00, C01
    
    # P6
    U[1*5+0, 5] += 1; U[0*5+0, 5] -= 1
    V[0*5+0, 5] += 1; V[0*5+1, 5] += 1
    W[1*3+1, 5] += 1 # C11
    
    # P7
    U[0*5+1, 6] += 1; U[1*5+1, 6] -= 1
    V[1*5+0, 6] += 1; V[1*5+1, 6] += 1
    W[0*3+0, 6] -= 1; W[1*3+1, 6] -= 1 # C00, C11
    
    # 2. Schoolbook for the rest
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the Strassen block
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
