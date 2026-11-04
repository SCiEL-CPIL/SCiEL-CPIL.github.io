def add_strassen_block(U, V, W, i_base, j_base, k_base, start_col):
    n, m, p = 3, 4, 8
    # Local indices 0,1
    # Strassen M1..M7
    # u[i'][j'] coeffs for A
    # v[j'][k'] coeffs for B
    
    # M1: (A00+A11)(B00+B11)
    # u: [[1,0],[0,1]]
    # v: [[1,1],[1,1]]
    
    # M2: (A10+A11)B00
    # u: [[0,0],[1,1]]
    # v: [[1,0],[0,0]]
    
    # M3: A00(B01-B11)
    # u: [[1,0],[0,0]]
    # v: [[0,1],[0,-1]]
    
    # M4: A11(B10-B00)
    # u: [[0,0],[0,1]]
    # v: [[-1,0],[1,0]]
    
    # M5: (A00+A01)B11
    # u: [[1,1],[0,0]]
    # v: [[0,0],[0,1]]
    
    # M6: (A10-A00)(B00+B01)
    # u: [[-1,0],[1,0]]
    # v: [[1,1],[0,0]]
    
    # M7: (A01-A11)(B10+B11)
    # u: [[0,1],[-1,0]]
    # v: [[0,1],[1,1]]
    
    us = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[-1,0]]
    ]
    vs = [
        [[1,1],[1,1]],
        [[1,0],[0,0]],
        [[0,1],[0,-1]],
        [[-1,0],[1,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,1],[1,1]]
    ]
    
    # Reconstruction of C
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # w[k'][i'] is the coefficient of M_t in C[i'][k']?
    # No, C[i'][k'] = sum_t coeff[t][i'][k'] * M_t
    # So W index for output (i', k') gets coeff[t][i'][k'] for column start_col + t.
    
    coeffs = [
        [[1, 0], [0, 1]], # M1 contributes to C00 (1), C11 (1) -> C[i][k]
        # Wait, C[i'][k']
        # C00 (i'=0, k'=0): M1(1), M4(1), M5(-1), M7(1)
        # C01 (i'=0, k'=1): M3(1), M5(1)
        # C10 (i'=1, k'=0): M2(1), M4(1)
        # C11 (i'=1, k'=1): M1(1), M2(-1), M3(1), M6(1)
        
        # Let's make a 3D array coeff[t][i'][k']
        [[1, 0], [0, 1]], # M1: C00+=1, C11+=1. So coeff[0][0][0]=1, coeff[0][1][1]=1
        [[0, 0], [1, -1]], # M2: C10+=1, C11+=-1. coeff[1][1][0]=1, coeff[1][1][1]=-1
        [[0, 1], [0, 1]], # M3: C01+=1, C11+=1. coeff[2][0][1]=1, coeff[2][1][1]=1
        [[1, 0], [0, 0]], # M4: C00+=1, C10+=1. coeff[3][0][0]=1, coeff[3][1][0]=1
        [[-1, 1], [0, 0]], # M5: C00+=-1, C01+=1. coeff[4][0][0]=-1, coeff[4][0][1]=1
        [[0, 0], [0, 1]], # M6: C11+=1. coeff[5][1][1]=1
        [[1, 0], [0, 0]]  # M7: C00+=1. coeff[6][0][0]=1
    ]
    
    for t in range(7):
        col = start_col + t
        # Fill U
        for i_l in range(2):
            for j_l in range(2):
                i_g = i_base + i_l
                j_g = j_base + j_l
                U[i_g * m + j_g][col] += us[t][i_l][j_l]
        
        # Fill V
        for j_l in range(2):
            for k_l in range(2):
                j_g = j_base + j_l
                k_g = k_base + k_l
                V[j_g * p + k_g][col] += vs[t][j_l][k_l]
                
        # Fill W
        for i_l in range(2):
            for k_l in range(2):
                i_g = i_base + i_l
                k_g = k_base + k_l
                W[k_g * n + i_g][col] += coeffs[t][i_l][k_l]
                
    return start_col + 7
