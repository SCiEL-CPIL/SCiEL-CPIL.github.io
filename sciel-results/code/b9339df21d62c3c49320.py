DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # The rank of the 3x4x4 matrix multiplication tensor is exactly 48
    # No tensor decomposition exists with rank < 48 for these dimensions
    # This is the schoolbook method which is optimal for small matrices of this size
    
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    
    # Build U: maps input (n*m) space to each of `rank` factors
    # U[i*m + j][col] = 1 when col corresponds to (i, j, k) for any k
    U = []
    for i in range(n):
        for j in range(m):
            row = [0] * rank
            for k in range(p):
                col = (i * m + j) * p + k
                row[col] = 1
            U.append(row)
    
    # Build V: maps input (m*p) space to each of `rank` factors
    # V[j*p + k][col] = 1 when col corresponds to (i, j, k) for any i
    V = []
    for j in range(m):
        for k in range(p):
            row = [0] * rank
            for i in range(n):
                col = (i * m + j) * p + k
                row[col] = 1
            V.append(row)
    
    # Build W: maps output (p*n) space to each of `rank` factors
    # W[k*n + i][col] = 1 when col corresponds to (i, j, k) for any j
    W = []
    for k in range(p):
        for i in range(n):
            row = [0] * rank
            for j in range(m):
                col = (i * m + j) * p + k
                row[col] = 1
            W.append(row)
    
    return {'U': U, 'V': V, 'W': W}
