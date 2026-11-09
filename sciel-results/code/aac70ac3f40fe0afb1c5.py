def construct():
    n, m, p = 2, 5, 5
    rank = 46
    U = [[0]*rank for _ in range(n*m)]  # 10 x 46
    V = [[0]*rank for _ in range(m*p)]  # 25 x 46
    W = [[0]*rank for _ in range(p*n)]  # 10 x 46
    
    c = 0  # current column index
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal c
        # X = A[i_off:i_off+2, j_off:j_off+2]
        # Y = B[j_off:j_off+2, k_off:k_off+2]
        # Z = C[i_off:i_off+2, k_off:k_off+2]
        
        # X entries: X[0,0]=A[i_off,j_off], X[0,1]=A[i_off,j_off+1], X[1,0]=A[i_off+1,j_off], X[1,1]=A[i_off+1,j_off+1]
        # Y entries: Y[0,0]=B[j_off,k_off], Y[0,1]=B[j_off,k_off+1], Y[1,0]=B[j_off+1,k_off], Y[1,1]=B[j_off+1,k_off+1]
        # Z entries: Z[0,0]=C[i_off,k_off], Z[0,1]=C[i_off,k_off+1], Z[1,0]=C[i_off+1,k_off], Z[1,1]=C[i_off+1,k_off+1]
        
        # Strassen factors:
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Output:
        # Z00 = M1+M4-M5+M7
        # Z01 = M3+M5
        # Z10 = M2+M4
        # Z11 = M1-M2+M3+M6
        
        # U indices for X:
        # X[0,0] = A[i_off, j_off] → U[i_off*m + j_off]
        # X[0,1] = A[i_off, j_off+1] → U[i_off*m + j_off+1]
        # X[1,0] = A[i_off+1, j_off] → U[(i_off+1)*m + j_off]
        # X[1,1] = A[i_off+1, j_off+1] → U[(i_off+1)*m + j_off+1]
        
        # V indices for Y:
        # Y[0,0] = B[j_off, k_off] → V[j_off*p + k_off]
        # Y[0,1] = B[j_off, k_off+1] → V[j_off*p + k_off+1]
        # Y[1,0] = B[j_off+1, k_off] → V[(j_off+1)*p + k_off]
        # Y[1,1] = B[j_off+1, k_off+1] → V[(j_off+1)*p + k_off+1]
        
        # W indices for Z:
        # Z[0,0] = C[i_off, k_off] → W[k_off*n + i_off]
        # Z[0,1] = C[i_off, k_off+1] → W[(k_off+1)*n + i_off]
        # Z[1,0] = C[i_off+1, k_off] → W[k_off*n + (i_off+1)]
        # Z[1,1] = C[i_off+1, k_off+1] → W[(k_off+1)*n + (i_off+1)]
        
        x00 = i_off * m + j_off
        x01 = i_off * m + j_off + 1
        x10 = (i_off+1) * m + j_off
        x11 = (i_off+1) * m + j_off + 1
        
        y00 = j_off * p + k_off
        y01 = j_off * p + k_off + 1
        y10 = (j_off+1) * p + k_off
        y11 = (j_off+1) * p + k_off + 1
        
        z00 = k_off * n + i_off
        z01 = (k_off+1) * n + i_off
        z10 = k_off * n + (i_off+1)
        z11 = (k_off+1) * n + (i_off+1)
        
        # M1 = (X00+X11)(Y00+Y11)
        U[x00][c] += 1; U[x11][c] += 1
        V[y00][c] += 1; V[y11][c] += 1
        W[z00][c] += 1; W[z11][c] += 1  # Z00 gets +M1, Z11 gets +M1
        c += 1
        
        # M2 = (X10+X11)Y00
        U[x10][c] += 1; U[x11][c] += 1
        V[y00][c] += 1
        W[z10][c] += 1; W[z11][c] -= 1  # Z10 gets +M2, Z11 gets -M2
        c += 1
        
        # M3 = X00(Y01-Y11)
        U[x00][c] += 1
        V[y01][c] += 1; V[y11][c] -= 1
        W[z01][c] += 1; W[z11][c] += 1  # Z01 gets +M3, Z11 gets +M3
        c += 1
        
        # M4 = X11(Y10-Y00)
        U[x11][c] += 1
        V[y10][c] += 1; V[y00][c] -= 1
        W[z00][c] += 1; W[z10][c] += 1  # Z00 gets +M4, Z10 gets +M4
        c += 1
        
        # M5 = (X00+X01)Y11
        U[x00][c] += 1; U[x01][c] += 1
        V[y11][c] += 1
        W[z00][c] -= 1; W[z01][c] += 1  # Z00 gets -M5, Z01 gets +M5
        c += 1
        
        # M6 = (X10-X00)(Y00+Y01)
        U[x10][c] += 1; U[x00][c] -= 1
        V[y00][c] += 1; V[y01][c] += 1
        W[z11][c] += 1  # Z11 gets +M6
        c += 1
        
        # M7 = (X01-X11)(Y10+Y11)
        U[x01][c] += 1; U[x11][c] -= 1
        V[y10][c] += 1; V[y11][c] += 1
        W[z00][c] += 1  # Z00 gets +M7
        c += 1
    
    def add_schoolbook_2x2_x_2x1(i_off, j_off, k_off):
        """A[i_off:i_off+2, j_off:j_off+2] (2x2) × B[j_off:j_off+2, k_off] (2x1) → C[i_off:i_off+2, k_off] (2x1)"""
        nonlocal c
        for ii in range(2):
            for jj in range(2):
                U[(i_off+ii)*m + (j_off+jj)][c] += 1
                V[(j_off+jj)*p + k_off][c] += 1
                W[k_off*n + (i_off+ii)][c] += 1
                c += 1
    
    def add_schoolbook_2x1_x_1x2(i_off, j_off, k_off):
        """A[i_off:i_off+2, j_off] (2x1) × B[j_off, k_off:k_off+2] (1x2) → C[i_off:i_off+2, k_off:k_off+2] (2x2)"""
        nonlocal c
        for ii in range(2):
            for kk in range(2):
                U[(i_off+ii)*m + j_off][c] += 1
                V[j_off*p + (k_off+kk)][c] += 1
                W[(k_off+kk)*n + (i_off+ii)][c] += 1
                c += 1
    
    def add_schoolbook_2x1_x_1x1(i_off, j_off, k_off):
        """A[i_off:i_off+2, j_off] (2x1) × B[j_off, k_off] (1x1) → C[i_off:i_off+2, k_off] (2x1)"""
        nonlocal c
        for ii in range(2):
            U[(i_off+ii)*m + j_off][c] += 1
            V[j_off*p + k_off][c] += 1
            W[k_off*n + (i_off+ii)][c] += 1
            c += 1
    
    # Block 1: A[:,{0,1}] × B[{0,1},{0,1}] → C[{0,1},{0,1}]
    add_strassen_block(0, 0, 0)
    
    # Block 2: A[:,{0,1}] × B[{0,1},{2,3}] → C[{0,1},{2,3}]
    add_strassen_block(0, 0, 2)
    
    # Block 3: A[:,{0,1}] × B[{0,1},{4}] → C[{0,1},{4}]
    add_schoolbook_2x2_x_2x1(0, 0, 4)
    
    # Block 4: A[:,{2,3}] × B[{2,3},{0,1}] → C[{0,1},{0,1}]
    add_strassen_block(0, 2, 0)
    
    # Block 5: A[:,{2,3}] × B[{2,3},{2,3}] → C[{0,1},{2,3}]
    add_strassen_block(0, 2, 2)
    
    # Block 6: A[:,{2,3}] × B[{2,3},{4}] → C[{0,1},{4}]
    add_schoolbook_2x2_x_2x1(0, 2, 4)
    
    # Block 7: A[:,{4}] × B[{4},{0,1}] → C[{0,1},{0,1}]
    add_schoolbook_2x1_x_1x2(0, 4, 0)
    
    # Block 8: A[:,{4}] × B[{4},{2,3}] → C[{0,1},{2,3}]
    add_schoolbook_2x1_x_1x2(0, 4, 2)
    
    # Block 9: A[:,{4}] × B[{4},{4}] → C[{0,1},{4}]
    add_schoolbook_2x1_x_1x1(0, 4, 4)
    
    return {"U": U, "V": V, "W": W}
