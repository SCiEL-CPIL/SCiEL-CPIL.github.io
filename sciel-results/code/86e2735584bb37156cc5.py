import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Must finish within 120s, keep safety margin
    
    n, m, p = 4, 5, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    # 8 Strassen blocks (rank 7 each) + 3 Slab terms (rank 1 each)
    rank = 8 * 7 + 3
    # rank = 59
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Define all 2x2x2 blocks
    blocks = []
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2]:
                blocks.append((
                    [i_base, i_base + 1],
                    [j_base, j_base + 1],
                    [k_base, k_base + 1]
                ))
    
    col = 0
    # Fill Strassen terms for each block
    for I, J, K in blocks:
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # M1 = (a11 + a22)(b11 + b22)
        U[i1 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[i2 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        W[k1 * n + i2][col] += 1
        W[k2 * n + i2][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[i1 * m + j1][col] += 1
        V[j1 * p + k2][col] += 1
        V[j2 * p + k2][col] += -1
        W[k2 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[i2 * m + j2][col] += 1
        V[j2 * p + k1][col] += 1
        V[j1 * p + k1][col] += -1
        W[k1 * n + i1][col] += 1
        W[k1 * n + i2][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[i1 * m + j1][col] += 1
        U[i1 * m + j2][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += -1
        W[k2 * n + i1][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[i2 * m + j1][col] += 1
        U[i1 * m + j1][col] += -1
        V[j1 * p + k1][col] += 1
        V[j1 * p + k2][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[i1 * m + j2][col] += 1
        U[i2 * m + j2][col] += -1
        V[j2 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        col += 1
        
    # Add Slab terms for the remainder (j=4 or k=4)
    # Remainder R = { (i,j,k) | j=4 or k=4 }
    # T_R = T_{j=4} + T_{k=4} - T_{j=4, k=4}
    
    # Term 1: T_{j=4} = (sum_i e_i) (e_4) (sum_k e_k)
    # U[i*m + 4, col] = 1 for all i
    # V[4*p + k, col] = 1 for all k
    # W[k*n + i, col] = 1 for all k, i
    
    for i in range(n):
        U[i * m + 4][col] += 1
    for k in range(p):
        V[4 * p + k][col] += 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] += 1
    col += 1
    
    # Term 2: T_{k=4} = (sum_i e_i) (sum_j e_j) (e_4)
    # U[i*m + j, col] = 1 for all i, j
    # V[j*p + 4, col] = 1 for all j
    # W[4*n + i, col] = 1 for all i
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] += 1
    for j in range(m):
        V[j * p + 4][col] += 1
    for i in range(n):
        W[4 * n + i][col] += 1
    col += 1
    
    # Term 3: -T_{j=4, k=4} = -(sum_i e_i) (e_4) (e_4)
    # U[i*m + 4, col] = -1 for all i
    # V[4*p + 4, col] = 1
    # W[4*n + i, col] = 1 for all i
    
    for i in range(n):
        U[i * m + 4][col] += -1
    V[4 * p + 4][col] += 1
    for i in range(n):
        W[4 * n + i][col] += 1
    col += 1
    
    # Check if we're within time limit
    if time.time() > deadline:
        pass
    
    return {"U": U, "V": V, "W": W}
