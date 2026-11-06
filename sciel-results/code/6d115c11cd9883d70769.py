def construct():
    n, m, p = 6, 6, 6
    rank = 63
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for Mi in range(7):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Determine A_factor, B_factor, C_contributions based on Mi
                    # A_factor is a dict mapping (row, col) -> coefficient
                    # B_factor is a dict mapping (row, col) -> coefficient
                    # C_contributions is a dict mapping (row, col) -> coefficient
                    
                    A_factor = {}
                    B_factor = {}
                    C_contrib = {}
                    
                    if Mi == 0:  # M1 = (A11+A22)*(B11+B22)
                        A_factor[(i,j)] = 1
                        A_factor[(3+i,3+j)] = 1
                        B_factor[(j,k)] = 1
                        B_factor[(3+j,3+k)] = 1
                        C_contrib[(i,k)] = 1
                        C_contrib[(3+i,3+k)] = 1
                    elif Mi == 1:  # M2 = (A21+A22)*B11
                        A_factor[(3+i,j)] = 1
                        A_factor[(3+i,3+j)] = 1
                        B_factor[(j,k)] = 1
                        C_contrib[(3+i,k)] = 1
                        C_contrib[(3+i,3+k)] = -1
                    elif Mi == 2:  # M3 = A11*(B12-B22)
                        A_factor[(i,j)] = 1
                        B_factor[(j,3+k)] = 1
                        B_factor[(3+j,3+k)] = -1
                        C_contrib[(i,3+k)] = 1
                        C_contrib[(3+i,3+k)] = 1
                    elif Mi == 3:  # M4 = A22*(B21-B11)
                        A_factor[(3+i,3+j)] = 1
                        B_factor[(3+j,k)] = 1
                        B_factor[(j,k)] = -1
                        C_contrib[(i,k)] = 1
                        C_contrib[(3+i,k)] = 1
                    elif Mi == 4:  # M5 = (A11+A12)*B22
                        A_factor[(i,j)] = 1
                        A_factor[(i,3+j)] = 1
                        B_factor[(3+j,3+k)] = 1
                        C_contrib[(i,k)] = -1
                        C_contrib[(i,3+k)] = 1
                    elif Mi == 5:  # M6 = (A21-A11)*(B11+B12)
                        A_factor[(3+i,j)] = 1
                        A_factor[(i,j)] = -1
                        B_factor[(j,k)] = 1
                        B_factor[(j,3+k)] = 1
                        C_contrib[(3+i,3+k)] = 1
                    elif Mi == 6:  # M7 = (A12-A22)*(B21+B22)
                        A_factor[(i,3+j)] = 1
                        A_factor[(3+i,3+j)] = -1
                        B_factor[(3+j,k)] = 1
                        B_factor[(3+j,3+k)] = 1
                        C_contrib[(i,k)] = 1
                    
                    # Fill U, V, W
                    for (ar, ac), coeff in A_factor.items():
                        U[ar*m+ac][col] += coeff
                    for (br, bc), coeff in B_factor.items():
                        V[br*p+bc][col] += coeff
                    for (cr, cc), coeff in C_contrib.items():
                        # W is indexed by cc*n+cr
                        W[cc*n+cr][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
