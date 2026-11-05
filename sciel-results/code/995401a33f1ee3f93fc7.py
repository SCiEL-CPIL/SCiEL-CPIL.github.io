"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7."""
    # Strassen: 7 multiplications
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # i,j,k in {0,1}
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    V = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    W = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    
    # Index: U[i*2+j], V[j*2+k], W[k*2+i]
    # i,j,k: 0 or 1
    
    # M1: (a11+a22)(b11+b22) → U: a11,a22; V: b11,b22; W: c11(+), c22(+)
    U[0][0] = 1  # a11
    U[3][0] = 1  # a22
    V[0][0] = 1  # b11
    V[3][0] = 1  # b22
    W[0][0] = 1  # c11 (k=0,i=0)
    W[3][0] = 1  # c22 (k=1,i=1)
    
    # M2: (a21+a22)b11 → U: a21,a22; V: b11; W: c21(+)
    U[2][1] = 1  # a21
    U[3][1] = 1  # a22
    V[0][1] = 1  # b11
    W[2][1] = 1  # c21 (k=1,i=0)
    
    # M3: a11(b12-b22) → U: a11; V: b12(-),b22; W: c12(+), c22(+)
    U[0][2] = 1  # a11
    V[1][2] = 1  # b12
    V[3][2] = -1  # b22
    W[1][2] = 1  # c12 (k=0,i=1)
    W[3][2] = 1  # c22 (k=1,i=1)
    
    # M4: a22(b21-b11) → U: a22; V: b21,b11(-); W: c11(+), c21(+)
    U[3][3] = 1  # a22
    V[2][3] = 1  # b21
    V[0][3] = -1  # b11
    W[0][3] = 1  # c11 (k=0,i=0)
    W[2][3] = 1  # c21 (k=1,i=0)
    
    # M5: (a11+a12)b22 → U: a11,a12; V: b22; W: c11(-), c12(+)
    U[0][4] = 1  # a11
    U[1][4] = 1  # a12
    V[3][4] = 1  # b22
    W[0][4] = -1  # c11
    W[1][4] = 1  # c12
    
    # M6: (a21-a11)(b11+b12) → U: a21,a11(-); V: b11,b12; W: c22(+)
    U[2][5] = 1  # a21
    U[0][5] = -1  # a11
    V[0][5] = 1  # b11
    V[1][5] = 1  # b12
    W[3][5] = 1  # c22
    
    # M7: (a12-a22)(b21+b22) → U: a12,a22(-); V: b21,b22; W: c11(+)
    U[1][6] = 1  # a12
    U[3][6] = -1  # a22
    V[2][6] = 1  # b21
    V[3][6] = 1  # b22
    W[0][6] = 1  # c11
    
    return U, V, W

def _schoolbook_block(ni, mj, pk):
    """Return (U, V, W) for schoolbook ni x mj x pk, rank ni*mj*pk."""
    rank = ni * mj * pk
    U = [[0]*rank for _ in range(ni*mj)]
    V = [[0]*rank for _ in range(mj*pk)]
    W = [[0]*rank for _ in range(pk*ni)]
    col = 0
    for i in range(ni):
        for j in range(mj):
            for k in range(pk):
                U[i*mj+j][col] = 1
                V[j*pk+k][col] = 1
                W[k*ni+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    # Split: n=[2,1], m=[2,2,1], p=[2,2,2]
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    for ni in n_splits:
        for mj in m_splits:
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    bu, bv, bw = _strassen_2x2x2()
                else:
                    bu, bv, bw = _schoolbook_block(ni, mj, pk)
                U_rows.append(bu)
                V_rows.append(bv)
                W_rows.append(bw)
    
    # Concatenate
    U = [row for block in U_rows for row in block]
    V = [row for block in V_rows for row in block]
    W = [row for block in W_rows for row in block]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
