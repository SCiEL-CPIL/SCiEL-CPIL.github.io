def construct():
    n, m, p = 2, 3, 4
    rank = n * m * p
    
    # Generate indices for U, V, W
    # U is 6x24, V is 12x24, W is 8x24
    # We can create the matrices by iterating through the rank dimension
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The schoolbook decomposition corresponds to the basis tensors
    # T_{i,j,k} = e_{ij} \otimes e_{jk} \otimes e_{ik}
    # In the flattened interface:
    # U[row, col] = 1 if row == i*m+j and col == i*m*p + j*p + k
    # V[row, col] = 1 if row == j*p+k and col == i*m*p + j*p + k
    # W[row, col] = 1 if row == k*n+i and col == i*m*p + j*p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                
    return {"U": U, "V": V, "W": W}
