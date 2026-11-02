DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # 6 = 3 * 2.
    # 3x3 blocks of 2x2.
    # 9 positions.
    # Each position is sum of 3 products of 2x2 matrices.
    # Total 27 products.
    # Each 2x2 product has rank 4.
    # Total rank 108.
    
    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i1 in range(3):
        for j1 in range(3):
            for k1 in range(3):
                # This is a term in the outer sum: A[i1, k1] * B[k1, j1]
                # Wait, C[i1, j1] = sum_k1 A[i1, k1] B[k1, j1]
                # So for each i1, j1, k1, we have a product of 2x2 matrices.
                # We decompose this 2x2 product using schoolbook (rank 4).
                
                # 2x2 schoolbook terms: (i2, j2, k2) in {0,1}^3
                for i2 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            # Term: A[i1*2+i2, k1*2+k2] * B[k1*2+k2, j1*2+j2] -> C[i1*2+i2, j1*2+j2]
                            # U index: row = i1*2+i2, col = k1*2+k2
                            # V index: row = k1*2+k2, col = j1*2+j2
                            # W index: row = j1*2+j2, col = i1*2+i2
                            
                            u_idx = (i1 * 2 + i2) * m + (k1 * 2 + k2)
                            v_idx = (k1 * 2 + k2) * p + (j1 * 2 + j2)
                            w_idx = (j1 * 2 + j2) * n + (i1 * 2 + i2)
                            
                            U[u_idx][col] = 1
                            V[v_idx][col] = 1
                            W[w_idx][col] = 1
                            col += 1
                            
    return {"U": U, "V": V, "W": W}
