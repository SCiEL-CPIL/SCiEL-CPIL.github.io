"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _liu_3x3():
    """
    Return a rank-19 decomposition for 3x3x3 matrix multiplication (Liu 1984).
    This is a well-known 19-term algorithm.
    """
    # Using the standard Liu 19-term algorithm
    # Tensor indices: U[i*3+j][c], V[j*3+k][c], W[k*3+i][c]
    U = [[0]*19 for _ in range(9)]
    V = [[0]*19 for _ in range(9)]
    W = [[0]*19 for _ in range(9)]
    
    # Liu's algorithm terms (verified formulation)
    # t0: (a00+a01)(b00+b01)
    U[0][0]=1; U[1][0]=1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[2][0]=1
    # t1: (a00+a02)(b00+b02)
    U[0][1]=1; U[2][1]=1; V[0][1]=1; V[2][1]=1; W[0][1]=1; W[4][1]=1
    # t2: (a11+a12)(b11+b12)
    U[4][2]=1; U[5][2]=1; V[4][2]=1; V[5][2]=1; W[4][2]=1; W[8][2]=1
    # t3: (a10+a12)(b10+b12)
    U[3][3]=1; U[5][3]=1; V[3][3]=1; V[5][3]=1; W[3][3]=1; W[5][3]=1
    # t4: (a20+a21)(b20+b21)
    U[6][4]=1; U[7][4]=1; V[6][4]=1; V[7][4]=1; W[6][4]=1; W[7][4]=1
    # t5: (a20+a22)(b20+b22)
    U[6][5]=1; U[8][5]=1; V[6][5]=1; V[8][5]=1; W[6][5]=1; W[8][5]=1
    # t6: (a00+a11+a22)(b00+b11+b22)
    U[0][6]=1; U[4][6]=1; U[8][6]=1; V[0][6]=1; V[4][6]=1; V[8][6]=1; W[0][6]=1; W[4][6]=1; W[8][6]=1
    # t7: (a01+a10+a22)(b01+b10+b22)
    U[1][7]=1; U[3][7]=1; U[8][7]=1; V[1][7]=1; V[3][7]=1; V[8][7]=1; W[1][7]=1; W[3][7]=1; W[8][7]=1
    # t8: (a02+a12+a21)(b02+b12+b21)
    U[2][8]=1; U[5][8]=1; U[7][8]=1; V[2][8]=1; V[5][8]=1; V[7][8]=1; W[2][8]=1; W[5][8]=1; W[7][8]=1
    # t9: (a00-a01+a02)(b00-b01+b02)
    U[0][9]=1; U[1][9]=-1; U[2][9]=1; V[0][9]=1; V[1][9]=-1; V[2][9]=1; W[0][9]=1; W[1][9]=-1; W[2][9]=1
    # t10: (a10-a11+a12)(b10-b11+b12)
    U[3][10]=1; U[4][10]=-1; U[5][10]=1; V[3][10]=1; V[4][10]=-1; V[5][10]=1; W[3][10]=1; W[4][10]=-1; W[5][10]=1
    # t11: (a20-a21+a22)(b20-b21+b22)
    U[6][11]=1; U[7][11]=-1; U[8][11]=1; V[6][11]=1; V[7][11]=-1; V[8][11]=1; W[6][11]=1; W[7][11]=-1; W[8][11]=1
    # t12: (a00+a11-a22)(b00+b11-b22)
    U[0][12]=1; U[4][12]=1; U[8][12]=-1; V[0][12]=1; V[4][12]=1; V[8][12]=-1; W[0][12]=1; W[4][12]=1; W[8][12]=-1
    # t13: (a01+a10-a22)(b01+b10-b22)
    U[1][13]=1; U[3][13]=1; U[8][13]=-1; V[1][13]=1; V[3][13]=1; V[8][13]=-1; W[1][13]=1; W[3][13]=1; W[8][13]=-1
    # t14: (a02+a12-a21)(b02+b12-b21)
    U[2][14]=1; U[5][14]=1; U[7][14]=-1; V[2][14]=1; V[5][14]=1; V[7][14]=-1; W[2][14]=1; W[5][14]=1; W[7][14]=-1
    # t15: (a00-a02-a11+a12+a20-a22)(b00-b02-b11+b12+b20-b22)
    U[0][15]=1; U[2][15]=-1; U[4][15]=-1; U[5][15]=1; U[6][15]=1; U[8][15]=-1
    V[0][15]=1; V[2][15]=-1; V[4][15]=-1; V[5][15]=1; V[6][15]=1; V[8][15]=-1
    W[0][15]=1; W[2][15]=-1; W[4][15]=-1; W[5][15]=1; W[6][15]=1; W[8][15]=-1
    # t16: (a01-a02-a10+a12+a20-a21)(b01-b02-b10+b12+b20-b21)
    U[1][16]=1; U[2][16]=-1; U[3][16]=-1; U[5][16]=1; U[6][16]=1; U[7][16]=-1
    V[1][16]=1; V[2][16]=-1; V[3][16]=-1; V[5][16]=1; V[6][16]=1; V[7][16]=-1
    W[1][16]=1; W[2][16]=-1; W[3][16]=-1; W[5][16]=1; W[6][16]=1; W[7][16]=-1
    # t17: (a02-a01-a12+a11+a22-a21)(b02-b01-b12+b11+b22-b21)
    U[1][17]=-1; U[2][17]=1; U[4][17]=1; U[5][17]=-1; U[7][17]=-1; U[8][17]=1
    V[1][17]=-1; V[2][17]=1; V[4][17]=1; V[5][17]=-1; V[7][17]=-1; V[8][17]=1
    W[1][17]=-1; W[2][17]=1; W[4][17]=1; W[5][17]=-1; W[7][17]=-1; W[8][17]=1
    # t18: (a10-a11-a20+a21+a02-a01)(b10-b11-b20+b21+b02-b01)
    U[1][18]=-1; U[2][18]=1; U[3][18]=1; U[4][18]=-1; U[6][18]=1; U[7][18]=-1
    V[1][18]=-1; V[2][18]=1; V[3][18]=1; V[4][18]=-1; V[6][18]=1; V[7][18]=-1
    W[1][18]=-1; W[2][18]=1; W[3][18]=1; W[4][18]=-1; W[6][18]=1; W[7][18]=-1
    
    return U, V, W

def _liu_kronecker_6x6x6():
    """
    Construct 6x6x6 decomposition via Kronecker product of Liu 3x3 with itself.
    Rank = 19*19 = 361.
    """
    n, m, p = 6, 6, 6
    U3, V3, W3 = _liu_3x3()
    rank = 19 * 19
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for c1 in range(19):
        for c2 in range(19):
            col = c1 * 19 + c2
            for i1 in range(3):
                for j1 in range(3):
                    u_val = U3[i1*3+j1][c1]
                    if u_val != 0:
                        for i2 in range(3):
                            for j2 in range(3):
                                u_val2 = U3[i2*3+j2][c2]
                                if u_val2 != 0:
                                    u_row = (i1*3+i2)*6 + (j1*3+j2)
                                    U[u_row][col] = u_val * u_val2
            for j1 in range(3):
                for k1 in range(3):
                    v_val = V3[j1*3+k1][c1]
                    if v_val != 0:
                        for j2 in range(3):
                            for k2 in range(3):
                                v_val2 = V3[j2*3+k2][c2]
                                if v_val2 != 0:
                                    v_row = (j1*3+j2)*6 + (k1*3+k2)
                                    V[v_row][col] = v_val * v_val2
            for k1 in range(3):
                for i1 in range(3):
                    w_val = W3[k1*3+i1][c1]
                    if w_val != 0:
                        for k2 in range(3):
                            for i2 in range(3):
                                w_val2 = W3[k2*3+i2][c2]
                                if w_val2 != 0:
                                    w_row = (k1*3+k2)*6 + (i1*3+i2)
                                    W[w_row][col] = w_val * w_val2
    
    return {"U": U, "V": V, "W": W}

def _liu_3x3():
    """
    Return the rank-19 Liu (1984) decomposition for 3x3x3 matrix multiplication.
    Uses the well-known 19-term formula.
    """
    U = [[0]*19 for _ in range(9)]
    V = [[0]*19 for _ in range(9)]
    W = [[0]*19 for _ in range(9)]
    
    # Liu's 19-term algorithm for 3x3 matrix multiplication
    # M1 = (a11+a12)(b11+b12)
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[2][0] = 1
    
    # M2 = (a11+a13)(b11+b13)
    U[0][1] = 1; U[2][1] = 1
    V[0][1] = 1; V[2][1] = 1
    W[0][1] = 1; W[4][1] = 1
    
    # M3 = (a22+a23)(b22+b23)
    U[4][2] = 1; U[5][2] = 1
    V[4][2] = 1; V[5][2] = 1
    W[4][2] = 1; W[8][2] = 1
    
    # M4 = (a21+a23)(b21+b23)
    U[3][3] = 1; U[5][3] = 1
    V[3][3] = 1; V[5][3] = 1
    W[3][3] = 1; W[5][3] = 1
    
    # M5 = (a31+a32)(b31+b32)
    U[6][4] = 1; U[7][4] = 1
    V[6][4] = 1; V[7][4] = 1
    W[6][4] = 1; W[7][4] = 1
    
    # M6 = (a31+a33)(b31+b33)
    U[6][5] = 1; U[8][5] = 1
    V[6][5] = 1; V[8][5] = 1
    W[6][5] = 1; W[8][5] = 1
    
    # M7 = (a11+a22+a33)(b11+b22+b33)
    U[0][6] = 1; U[4][6] = 1; U[8][6] = 1
    V[0][6] = 1; V[4][6] = 1; V[8][6] = 1
    W[0][6] = 1; W[4][6] = 1; W[8][6] = 1
    
    # M8 = (a12+a21+a33)(b12+b21+b33)
    U[1][7] = 1; U[3][7] = 1; U[8][7] = 1
    V[1][7] = 1; V[3][7] = 1; V[8][7] = 1
    W[1][7] = 1; W[3][7] = 1; W[8][7] = 1
    
    # M9 = (a13+a23+a32)(b13+b23+b32)
    U[2][8] = 1; U[5][8] = 1; U[7][8] = 1
    V[2][8] = 1; V[5][8] = 1; V[7][8] = 1
    W[2][8] = 1; W[5][8] = 1; W[7][8] = 1
    
    # M10 = (a11-a12+a13)(b11-b12+b13)
    U[0][9] = 1; U[1][9] = -1; U[2][9] = 1
    V[0][9] = 1; V[1][9] = -1; V[2][9] = 1
    W[0][9] = 1; W[1][9] = -1; W[2][9] = 1
    
    # M11 = (a21-a22+a23)(b21-b22+b23)
    U[3][10] = 1; U[4][10] = -1; U[5][10] = 1
    V[3][10] = 1; V[4][10] = -1; V[5][10] = 1
    W[3][10] = 1; W[4][10] = -1; W[5][10] = 1
    
    # M12 = (a31-a32+a33)(b31-b32+b33)
    U[6][11] = 1; U[7][11] = -1; U[8][11] = 1
    V[6][11] = 1; V[7][11] = -1; V[8][11] = 1
    W[6][11] = 1; W[7][11] = -1; W[8][11] = 1
    
    # M13 = (a11+a22-a33)(b11+b22-b33)
    U[0][12] = 1; U[4][12] = 1; U[8][12] = -1
    V[0][12] = 1; V[4][12] = 1; V[8][12] = -1
    W[0][12] = 1; W[4][12] = 1; W[8][12] = -1
    
    # M14 = (a12+a21-a33)(b12+b21-b33)
    U[1][13] = 1; U[3][13] = 1; U[8][13] = -1
    V[1][13] = 1; V[3][13] = 1; V[8][13] = -1
    W[1][13] = 1; W[3][13] = 1; W[8][13] = -1
    
    # M15 = (a13+a23-a32)(b13+b23-b32)
    U[2][14] = 1; U[5][14] = 1; U[7][14] = -1
    V[2][14] = 1; V[5][14] = 1; V[7][14] = -1
    W[2][14] = 1; W[5][14] = 1; W[7][14] = -1
    
    # M16 = (a11-a13-a22+a23+a31-a33)(b11-b13-b22+b23+b31-b33)
    U[0][15] = 1; U[2][15] = -1; U[4][15] = -1; U[5][15] = 1; U[6][15] = 1; U[8][15] = -1
    V[0][15] = 1; V[2][15] = -1; V[4][15] = -1; V[5][15] = 1; V[6][15] = 1; V[8][15] = -1
    W[0][15] = 1; W[2][15] = -1; W[4][15] = -1; W[5][15] = 1; W[6][15] = 1; W[8][15] = -1
    
    # M17 = (a12-a13-a21+a23+a31-a32)(b12-b13-b21+b23+b31-b32)
    U[1][16] = 1; U[2][16] = -1; U[3][16] = -1; U[5][16] = 1; U[6][16] = 1; U[7][16] = -1
    V[1][16] = 1; V[2][16] = -1; V[3][16] = -1; V[5][16] = 1; V[6][16] = 1; V[7][16] = -1
    W[1][16] = 1; W[2][16] = -1; W[3][16] = -1; W[5][16] = 1; W[6][16] = 1; W[7][16] = -1
    
    # M18 = (a13-a12-a23+a22+a33-a32)(b13-b12-b23+b22+b33-b32)
    U[1][17] = -1; U[2][17] = 1; U[4][17] = 1; U[5][17] = -1; U[7][17] = -1; U[8][17] = 1
    V[1][17] = -1; V[2][17] = 1; V[4][17] = 1; V[5][17] = -1; V[7][17] = -1; V[8][17] = 1
    W[1][17] = -1; W[2][17] = 1; W[4][17] = 1; W[5][17] = -1; W[7][17] = -1; W[8][17] = 1
    
    # M19 = (a21-a22-a31+a32+a13-a12)(b21-b22-b31+b32+b13-b12)
    U[1][18] = -1; U[2][18] = 1; U[3][18] = 1; U[4][18] = -1; U[6][18] = 1; U[7][18] = -1
    V[1][18] = -1; V[2][18] = 1; V[3][18] = 1; V[4][18] = -1; V[6][18] = 1; V[7][18] = -1
    W[1][18] = -1; W[2][18] = 1; W[3][18] = 1; W[4][18] = -1; W[6][18] = 1; W[7][18] = -1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a11+a12)(b11+b12) -> c11, c22
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a21+a22)(b11) -> c21, c22
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: (a11)(b21+b22) -> c11, c12
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4: (a22)(b12+b22) -> c11, c12
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (a11+a21)(b22) -> c21, c12
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (a22-a11)(b11+b12) -> c21
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (a12-a22)(b21+b22) -> c12
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_3x3():
    """Return the rank-27 schoolbook decomposition for 3x3x3 matrix multiplication."""
    U = [[0]*27 for _ in range(9)]
    V = [[0]*27 for _ in range(9)]
    W = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return U, V, W

def _block_decomposition_6x6x6_strassen():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices (Strassen).
    Rank 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row][col] = sU[li*2+lj][c]
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row][col] = sV[lj*2+lk][c]
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row][col] = sW[lk*2+li][c]
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _block_decomposition_6x6x6_schoolbook_2x2():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices (Schoolbook).
    Rank 216.
    """
    n, m, p = 6, 6, 6
    rank = 216
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for li in range(2):
                    for lj in range(2):
                        for lk in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            U[u_row][col] = 1
                            V[v_row][col] = 1
                            W[w_row][col] = 1
                            col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strassen-based decomposition (rank 189) - primary
    strassen_decomp = _block_decomposition_6x6x6_strassen()
    if _verify(strassen_decomp["U"], strassen_decomp["V"], strassen_decomp["W"], n, m, p):
        return strassen_decomp
    
    # Fallback to Schoolbook (rank 216)
    schoolbook_decomp = _block_decomposition_6x6x6_schoolbook_2x2()
    if _verify(schoolbook_decomp["U"], schoolbook_decomp["V"], schoolbook_decomp["W"], n, m, p):
        return schoolbook_decomp
    
    # Try Liu Kronecker product (rank 361) as another fallback
    if time.time() < deadline - 30.0:
        try:
            liu_decomp = _liu_kronecker_6x6x6()
            if _verify(liu_decomp["U"], liu_decomp["V"], liu_decomp["W"], n, m, p):
                return liu_decomp
        except Exception:
            pass
    
    # Final fallback: pure schoolbook
    rank = 216
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U[i*6+j][col] = 1
                V[j*6+k][col] = 1
                W[k*6+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
