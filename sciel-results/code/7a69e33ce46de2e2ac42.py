<<<<<<< SEARCH
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition, rank 7."""
        rank = 7
        U = [[0]*rank for _ in range(4)]  # 2x2 = 4 rows
        V = [[0]*rank for _ in range(4)]  # 2x2 = 4 rows
        W = [[0]*rank for _ in range(4)]  # 2x2 = 4 rows
        
        # M1 = (a00+a11)(b00+b11)
        # U: a00 and a11 contribute
        U[0][0] = 1  # a00
        U[3][0] = 1  # a11
        # V: b00 and b11 contribute
        V[0][0] = 1  # b00
        V[3][0] = 1  # b11
        # W: C00 += M1, C11 += M1
        W[0][0] = 1  # C00
        W[3][0] = 1  # C11
        
        # M2 = (a10+a11)b00
        U[2][1] = 1  # a10
        U[3][1] = 1  # a11
        V[0][1] = 1  # b00
        # W: C10 += M2, C11 += M2
        W[2][1] = 1  # C10
        W[3][1] = 1  # C11
        
        # M3 = a00(b01-b11)
        U[0][2] = 1  # a00
        V[1][2] = 1  # b01
        V[3][2] = -1 # -b11
        # W: C01 += M3, C11 -= M3
        W[1][2] = 1  # C01
        W[3][2] = -1 # C11
        
        # M4 = a11(b10-b00)
        U[3][3] = 1  # a11
        V[2][3] = 1  # b10
        V[0][3] = -1 # -b00
        # W: C00 -= M4, C10 += M4
        W[0][3] = -1 # C00
        W[2][3] = 1  # C10
        
        # M5 = (a00+a01)b11
        U[0][4] = 1  # a00
        U[1][4] = 1  # a01
        V[3][4] = 1  # b11
        # W: C01 += M5, C11 += M5
        W[1][4] = 1  # C01
        W[3][4] = 1  # C11
        
        # M6 = (a10-a00)(b00+b10)
        U[2][5] = 1  # a10
        U[0][5] = -1 # -a00
        V[0][5] = 1  # b00
        V[2][5] = 1  # b10
        # W: C00 += M6
        W[0][5] = 1  # C00
        
        # M7 = (a01-a11)(b01+b11)
        U[1][6] = 1  # a01
        U[3][6] = -1 # -a11
        V[1][6] = 1  # b01
        V[3][6] = 1  # b11
        # W: C00 += M7
        W[0][6] = 1  # C00
        
        return U, V, W
=======
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 schoolbook decomposition, rank 8."""
        rank = 8
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    c = i*2 + j*1 + k*0 # This is not unique, need to map to rank
                    # Schoolbook: each (i,j,k) is a separate rank term
                    # Rank index r = i*4 + j*2 + k
                    r = i*4 + j*2 + k
                    U[i*2+j][r] = 1
                    V[j*2+k][r] = 1
                    W[k*2+i][r] = 1
        return U, V, W
>>>>>>> REPLACE
