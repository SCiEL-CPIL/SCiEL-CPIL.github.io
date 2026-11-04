def construct():
    n, m, p = 4, 5, 6
    
    rank = 108
    U = [[0] * rank for _ in range(n * m)]  # (20, 108)
    V = [[0] * rank for _ in range(m * p)]  # (30, 108)
    W = [[0] * rank for _ in range(p * n)]  # (24, 108)
    
    col = 0
    
    # Strassen blocks for j in {0,1,2,3}
    for a in range(2):  # i-block: rows 2a, 2a+1
        for b in range(2):  # j-block: cols 2b, 2b+1
            for c in range(3):  # k-block: cols 2c, 2c+1
                # 7 Strassen terms
                # M1: (A11+A22)(B11+B22) -> C11+1, C22+1
                U[(2*a)*m + (2*b)][col] += 1
                U[(2*a+1)*m + (2*b+1)][col] += 1
                V[(2*b)*p + (2*c)][col] += 1
                V[(2*b+1)*p + (2*c+1)][col] += 1
                W[(2*c)*n + (2*a)][col] += 1
                W[(2*c+1)*n + (2*a+1)][col] += 1
                col += 1
                
                # M2: (A21+A22)B11 -> C21+1
                U[(2*a+1)*m + (2*b)][col] += 1
                U[(2*a+1)*m + (2*b+1)][col] += 1
                V[(2*b)*p + (2*c)][col] += 1
                W[(2*c)*n + (2*a+1)][col] += 1
                col += 1
                
                # M3: A11(B12-B22) -> C12+1, C22-1
                U[(2*a)*m + (2*b)][col] += 1
                V[(2*b)*p + (2*c+1)][col] += 1
                V[(2*b+1)*p + (2*c+1)][col] += -1
                W[(2*c+1)*n + (2*a)][col] += 1
                W[(2*c+1)*n + (2*a+1)][col] += -1
                col += 1
                
                # M4: A22(B21-B11) -> C21+1, C22-1
                U[(2*a+1)*m + (2*b+1)][col] += 1
                V[(2*b+1)*p + (2*c)][col] += 1
                V[(2*b)*p + (2*c)][col] += -1
                W[(2*c)*n + (2*a+1)][col] += 1
                W[(2*c+1)*n + (2*a+1)][col] += -1
                col += 1
                
                # M5: (A11+A12)B22 -> C11-1, C12+1
                U[(2*a)*m + (2*b)][col] += 1
                U[(2*a)*m + (2*b+1)][col] += 1
                V[(2*b+1)*p + (2*c+1)][col] += 1
                W[(2*c)*n + (2*a)][col] += -1
                W[(2*c+1)*n + (2*a)][col] += 1
                col += 1
                
                # M6: (A21-A11)(B11+B12) -> C22+1
                U[(2*a+1)*m + (2*b)][col] += 1
                U[(2*a)*m + (2*b)][col] += -1
                V[(2*b)*p + (2*c)][col] += 1
                V[(2*b)*p + (2*c+1)][col] += 1
                W[(2*c+1)*n + (2*a+1)][col] += 1
                col += 1
                
                # M7: (A12-A22)(B21+B22) -> C11+1, C22-1
                U[(2*a)*m + (2*b+1)][col] += 1
                U[(2*a+1)*m + (2*b+1)][col] += -1
                V[(2*b+1)*p + (2*c)][col] += 1
                V[(2*b+1)*p + (2*c+1)][col] += 1
                W[(2*c)*n + (2*a)][col] += 1
                W[(2*c+1)*n + (2*a+1)][col] += -1
                col += 1
    
    # Schoolbook for j=4 (5th middle index)
    for i in range(n):
        for k in range(p):
            U[i*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + i][col] += 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}
