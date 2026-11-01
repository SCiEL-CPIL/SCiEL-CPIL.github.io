"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Use Kronecker product structure to decompose 2x3 * 3x6 multiplication.
    
    Factorization: T_{2,3,6} = T_{2,3,2} ⊗ T_{1,1,3}
    where n1=2, m1=3, p1=2 and n2=1, m2=1, p2=3.
    
    The Kronecker product of two matrix multiplication tensors gives:
    T_{n1,m1,p1} ⊗ T_{n2,m2,p2} = T_{n1*n2, m1*m2, p1*p2}
    
    Index mapping:
    - i = i1*n2 + i2 (output row)
    - j = j1*m2 + j2 (middle index)  
    - k = k1*p2 + k2 (output column)
    
    For the tensor product, rank-1 components combine as:
    U[i, r] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
    V[j, r] = V1[j1*p1+k1, r1] * V2[j2*p2+k2, r2]
    W[k, r] = W1[k1*n1+i1, r1] * W2[k2*n2+i2, r2]
    where r = r1*r2 + r2
    """
    n, m, p = DIMENSIONS
    
    # Sub-problem dimensions
    n1, m1, p1 = 2, 3, 2  # First factor: 2x3 * 3x2
    n2, m2, p2 = 1, 1, 3  # Second factor: 1x1 * 1x3
    
    assert n1 * n2 == n
    assert m1 * m2 == m
    assert p1 * p2 == p
    
    # Schoolbook rank for sub-problem 1: T_{2,3,2}
    r1 = n1 * m1 * p1  # = 12
    # Schoolbook rank for sub-problem 2: T_{1,1,3}
    r2 = n2 * m2 * p2  # = 3
    
    rank = r1 * r2  # = 36
    
    # Build sub-problem 1: T_{n1,m1,p1}
    U1 = [[0] * r1 for _ in range(n1 * m1)]
    V1 = [[0] * r1 for _ in range(m1 * p1)]
    W1 = [[0] * r1 for _ in range(p1 * n1)]
    col = 0
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                U1[i1 * m1 + j1][col] = 1
                V1[j1 * p1 + k1][col] = 1
                W1[k1 * n1 + i1][col] = 1
                col += 1
    
    # Build sub-problem 2: T_{n2,m2,p2}
    U2 = [[0] * r2 for _ in range(n2 * m2)]
    V2 = [[0] * r2 for _ in range(m2 * p2)]
    W2 = [[0] * r2 for _ in range(p2 * n2)]
    col = 0
    for i2 in range(n2):
        for j2 in range(m2):
            for k2 in range(p2):
                U2[i2 * m2 + j2][col] = 1
                V2[j2 * p2 + k2][col] = 1
                W2[k2 * n2 + i2][col] = 1
                col += 1
    
    # Build global U, V, W via Kronecker product
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i, r] where i = (i1*n2+i2)*m + (j1*m2+j2), r = r1_idx*r2 + r2_idx
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            for i1 in range(n1):
                for j1 in range(m1):
                    val1 = U1[i1 * m1 + j1][r1_idx]
                    if val1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            val2 = U2[i2 * m2 + j2][r2_idx]
                            if val2 == 0:
                                continue
                            i_global = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                            U[i_global][r] = val1 * val2
    
    # V[j, r] where j = (j1*m2+j2)*p + (k1*p2+k2), r = r1_idx*r2 + r2_idx
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            for j1 in range(m1):
                for k1 in range(p1):
                    val1 = V1[j1 * p1 + k1][r1_idx]
                    if val1 == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            val2 = V2[j2 * p2 + k2][r2_idx]
                            if val2 == 0:
                                continue
                            j_global = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                            V[j_global][r] = val1 * val2
    
    # W[k, r] where k = (k1*p2+k2)*n + (i1*n2+i2), r = r1_idx*r2 + r2_idx
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            for k1 in range(p1):
                for i1 in range(n1):
                    val1 = W1[k1 * n1 + i1][r1_idx]
                    if val1 == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            val2 = W2[k2 * n2 + i2][r2_idx]
                            if val2 == 0:
                                continue
                            k_global = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                            W[k_global][r] = val1 * val2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
