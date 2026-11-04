def construct():
    n, m, p = 3, 4, 6
    i_groups = [[0, 1], [2]]
    j_groups = [[0, 1], [2, 3]]
    k_groups = [[0, 1], [2, 3], [4, 5]]
    
    U_entries = {}
    V_entries = {}
    W_entries = {}
    
    col = 0
    
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                if len(ig) == 2 and len(jg) == 2 and len(kg) == 2:
                    i0, i1 = ig
                    j0, j1 = jg
                    k0, k1 = kg
                    
                    # M1: (A00+A11)(B00+B11)
                    U_entries[(i0*m+j0, col)] = U_entries.get((i0*m+j0, col), 0) + 1
                    U_entries[(i1*m+j1, col)] = U_entries.get((i1*m+j1, col), 0) + 1
                    V_entries[(j0*p+k0, col)] = V_entries.get((j0*p+k0, col), 0) + 1
                    V_entries[(j1*p+k1, col)] = V_entries.get((j1*p+k1, col), 0) + 1
                    W_entries[(k0*n+i0, col)] = W_entries.get((k0*n+i0, col), 0) + 1
                    W_entries[(k1*n+i1, col)] = W_entries.get((k1*n+i1, col), 0) + 1
                    col += 1
                    
                    # M2: (A10+A11)B00
                    U_entries[(i1*m+j0, col)] = U_entries.get((i1*m+j0, col), 0) + 1
                    U_entries[(i1*m+j1, col)] = U_entries.get((i1*m+j1, col), 0) + 1
                    V_entries[(j0*p+k0, col)] = V_entries.get((j0*p+k0, col), 0) + 1
                    W_entries[(k0*n+i1, col)] = W_entries.get((k0*n+i1, col), 0) + 1
                    W_entries[(k1*n+i1, col)] = W_entries.get((k1*n+i1, col), 0) - 1
                    col += 1
                    
                    # M3: A00(B10-B11)
                    U_entries[(i0*m+j0, col)] = U_entries.get((i0*m+j0, col), 0) + 1
                    V_entries[(j1*p+k0, col)] = V_entries.get((j1*p+k0, col), 0) + 1
                    V_entries[(j1*p+k1, col)] = V_entries.get((j1*p+k1, col), 0) - 1
                    W_entries[(k0*n+i0, col)] = W_entries.get((k0*n+i0, col), 0) + 1
                    W_entries[(k1*n+i0, col)] = W_entries.get((k1*n+i0, col), 0) + 1
                    col += 1
                    
                    # M4: A11(B01-B00)
                    U_entries[(i1*m+j1, col)] = U_entries.get((i1*m+j1, col), 0) + 1
                    V_entries[(j0*p+k1, col)] = V_entries.get((j0*p+k1, col), 0) + 1
                    V_entries[(j0*p+k0, col)] = V_entries.get((j0*p+k0, col), 0) - 1
                    W_entries[(k0*n+i0, col)] = W_entries.get((k0*n+i0, col), 0) + 1
                    W_entries[(k0*n+i1, col)] = W_entries.get((k0*n+i1, col), 0) + 1
                    col += 1
                    
                    # M5: (A00+A01)B11
                    U_entries[(i0*m+j0, col)] = U_entries.get((i0*m+j0, col), 0) + 1
                    U_entries[(i0*m+j1, col)] = U_entries.get((i0*m+j1, col), 0) + 1
                    V_entries[(j1*p+k1, col)] = V_entries.get((j1*p+k1, col), 0) + 1
                    W_entries[(k1*n+i0, col)] = W_entries.get((k1*n+i0, col), 0) + 1
                    W_entries[(k0*n+i0, col)] = W_entries.get((k0*n+i0, col), 0) - 1
                    col += 1
                    
                    # M6: (A10-A00)(B00+B10)
                    U_entries[(i1*m+j0, col)] = U_entries.get((i1*m+j0, col), 0) + 1
                    U_entries[(i0*m+j0, col)] = U_entries.get((i0*m+j0, col), 0) - 1
                    V_entries[(j0*p+k0, col)] = V_entries.get((j0*p+k0, col), 0) + 1
                    V_entries[(j1*p+k0, col)] = V_entries.get((j1*p+k0, col), 0) + 1
                    W_entries[(k1*n+i1, col)] = W_entries.get((k1*n+i1, col), 0) + 1
                    col += 1
                    
                    # M7: (A01-A11)(B01+B11)
                    U_entries[(i0*m+j1, col)] = U_entries.get((i0*m+j1, col), 0) + 1
                    U_entries[(i1*m+j1, col)] = U_entries.get((i1*m+j1, col), 0) - 1
                    V_entries[(j0*p+k1, col)] = V_entries.get((j0*p+k1, col), 0) + 1
                    V_entries[(j1*p+k1, col)] = V_entries.get((j1*p+k1, col), 0) + 1
                    W_entries[(k0*n+i0, col)] = W_entries.get((k0*n+i0, col), 0) + 1
                    W_entries[(k1*n+i1, col)] = W_entries.get((k1*n+i1, col), 0) + 1
                    col += 1
                    
                elif len(ig) == 1 and len(jg) == 2 and len(kg) == 2:
                    i0 = ig[0]
                    j0, j1 = jg
                    k0, k1 = kg
                    
                    # A0*B00
                    U_entries[(i0*m+j0, col)] = U_entries.get((i0*m+j0, col), 0) + 1
                    V_entries[(j0*p+k0, col)] = V_entries.get((j0*p+k0, col), 0) + 1
                    W_entries[(k0*n+i0, col)] = W_entries.get((k0*n+i0, col), 0) + 1
                    col += 1
                    
                    # A1*B10
                    U_entries[(i0*m+j1, col)] = U_entries.get((i0*m+j1, col), 0) + 1
                    V_entries[(j1*p+k0, col)] = V_entries.get((j1*p+k0, col), 0) + 1
                    W_entries[(k0*n+i0, col)] = W_entries.get((k0*n+i0, col), 0) + 1
                    col += 1
                    
                    # A0*B01
                    U_entries[(i0*m+j0, col)] = U_entries.get((i0*m+j0, col), 0) + 1
                    V_entries[(j0*p+k1, col)] = V_entries.get((j0*p+k1, col), 0) + 1
                    W_entries[(k1*n+i0, col)] = W_entries.get((k1*n+i0, col), 0) + 1
                    col += 1
                    
                    # A1*B11
                    U_entries[(i0*m+j1, col)] = U_entries.get((i0*m+j1, col), 0) + 1
                    V_entries[(j1*p+k1, col)] = V_entries.get((j1*p+k1, col), 0) + 1
                    W_entries[(k1*n+i0, col)] = W_entries.get((k1*n+i0, col), 0) + 1
                    col += 1
    
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (row, c), val in U_entries.items():
        U[row][c] = val
    for (row, c), val in V_entries.items():
        V[row][c] = val
    for (row, c), val in W_entries.items():
        W[row][c] = val
    
    return {"U": U, "V": V, "W": W}
