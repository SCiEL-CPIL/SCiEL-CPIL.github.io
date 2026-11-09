def add_strassen(i0, i1, j0, j1, k0, k1):
    nonlocal col
    u00 = i0*m + j0
    u01 = i0*m + j1
    u10 = i1*m + j0
    u11 = i1*m + j1
    v00 = j0*p + k0
    v01 = j0*p + k1
    v10 = j1*p + k0
    v11 = j1*p + k1
    w00 = k0*n + i0  # C[i0,k0]
    w01 = k0*n + i1  # C[i1,k0]
    w10 = k1*n + i0  # C[i0,k1]
    w11 = k1*n + i1  # C[i1,k1]
    
    # p1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U[u00][col] += 1; U[u11][col] += 1
    V[v00][col] += 1; V[v11][col] += 1
    W[w00][col] += 1; W[w11][col] += 1
    col += 1
    
    # p2 = (A10+A11)B00 -> C10+=1
    U[u10][col] += 1; U[u11][col] += 1
    V[v00][col] += 1
    W[w01][col] += 1
    col += 1
    
    # p3 = A00(B01-B11) -> C01+=1
    U[u00][col] += 1
    V[v01][col] += 1; V[v11][col] += -1
    W[w10][col] += 1
    col += 1
    
    # p4 = A11(B10-B00) -> C00+=1, C10+=1
    U[u11][col] += 1
    V[v10][col] += 1; V[v00][col] += -1
    W[w00][col] += 1; W[w01][col] += 1
    col += 1
    
    # p5 = (A00+A01)B11 -> C00-=1, C01+=1
    U[u00][col] += 1; U[u01][col] += 1
    V[v11][col] += 1
    W[w00][col] += -1; W[w10][col] += 1
    col += 1
    
    # p6 = (A10-A00)(B00+B10) -> C11+=1
    U[u10][col] += 1; U[u00][col] += -1
    V[v00][col] += 1; V[v10][col] += 1
    W[w11][col] += 1
    col += 1
    
    # p7 = (A01-A11)(B01+B11) -> C01+=1, C11+=1
    U[u01][col] += 1; U[u11][col] += -1
    V[v01][col] += 1; V[v11][col] += 1
    W[w10][col] += 1; W[w11][col] += 1
    col += 1
