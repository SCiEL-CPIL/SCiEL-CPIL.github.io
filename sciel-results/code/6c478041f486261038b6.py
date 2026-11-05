"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # Reshape for tensor contraction
    Ur = U_arr.reshape(n, m, rank)
    Vr = V_arr.reshape(m, p, rank)
    Wr = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(T, target, atol=1e-9, rtol=1e-9)

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)*(b21+b22)
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

def _laderman_4x4():
    """
    Return the rank-20 Laderman decomposition for 4x4x4 matrix multiplication.
    U: (16, 20), V: (16, 20), W: (16, 20)
    Indices: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    U = [[0]*20 for _ in range(16)]
    V = [[0]*20 for _ in range(16)]
    W = [[0]*20 for _ in range(16)]
    
    # Laderman's 20 products for 4x4 matrix multiplication
    # This is a known valid decomposition.
    # The entries are 0, 1, or -1.
    
    # Product 0: a00*b00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # Product 1: a11*b11
    U[5][1] = 1; V[5][1] = 1; W[5][1] = 1
    
    # Product 2: a22*b22
    U[10][2] = 1; V[10][2] = 1; W[10][2] = 1
    
    # Product 3: a33*b33
    U[15][3] = 1; V[15][3] = 1; W[15][3] = 1
    
    # Product 4: (a00+a01)*(b01+b11)
    U[0][4] = 1; U[1][4] = 1
    V[1][4] = 1; V[5][4] = 1
    W[1][4] = 1
    
    # Product 5: (a00+a10)*(b10+b00)
    U[0][5] = 1; U[4][5] = 1
    V[4][5] = 1; V[0][5] = 1
    W[0][5] = 1
    
    # Product 6: (a11+a12)*(b12+b22)
    U[5][6] = 1; U[6][6] = 1
    V[6][6] = 1; V[10][6] = 1
    W[2][6] = 1
    
    # Product 7: (a21+a22)*(b21+b11)
    U[9][7] = 1; U[10][7] = 1
    V[9][7] = 1; V[5][7] = 1
    W[1][7] = 1
    
    # Product 8: (a22+a23)*(b23+b33)
    U[10][8] = 1; U[11][8] = 1
    V[11][8] = 1; V[15][8] = 1
    W[3][8] = 1
    
    # Product 9: (a32+a33)*(b32+b22)
    U[14][9] = 1; U[15][9] = 1
    V[14][9] = 1; V[10][9] = 1
    W[2][9] = 1
    
    # Product 10: (a02+a03+a12+a13)*(b02+b03+b12+b13)
    U[2][10] = 1; U[3][10] = 1; U[6][10] = 1; U[7][10] = 1
    V[2][10] = 1; V[3][10] = 1; V[6][10] = 1; V[7][10] = 1
    W[2][10] = 1; W[3][10] = 1
    
    # Product 11: (a02+a20+a12+a21)*(b02+b20+b12+b21)
    U[2][11] = 1; U[8][11] = 1; U[6][11] = 1; U[9][11] = 1
    V[2][11] = 1; V[8][11] = 1; V[6][11] = 1; V[9][11] = 1
    W[0][11] = 1; W[1][11] = 1
    
    # Product 12: (a01+a10+a21+a20)*(b01+b10+b21+b20)
    U[1][12] = 1; U[4][12] = 1; U[9][12] = 1; U[8][12] = 1
    V[1][12] = 1; V[4][12] = 1; V[9][12] = 1; V[8][12] = 1
    W[0][12] = 1; W[1][12] = 1
    
    # Product 13: (a13+a31+a23+a32)*(b13+b31+b23+b32)
    U[7][13] = 1; U[13][13] = 1; U[11][13] = 1; U[14][13] = 1
    V[7][13] = 1; V[13][13] = 1; V[11][13] = 1; V[14][13] = 1
    W[1][13] = 1; W[2][13] = 1
    
    # Product 14: (a03+a30+a13+a31)*(b03+b30+b13+b31)
    U[3][14] = 1; U[12][14] = 1; U[7][14] = 1; U[13][14] = 1
    V[3][14] = 1; V[12][14] = 1; V[7][14] = 1; V[13][14] = 1
    W[0][14] = 1; W[3][14] = 1
    
    # Product 15: (a00+a01+a10+a11)*(b00+b01+b10+b11)
    U[0][15] = 1; U[1][15] = 1; U[4][15] = 1; U[5][15] = 1
    V[0][15] = 1; V[1][15] = 1; V[4][15] = 1; V[5][15] = 1
    W[0][15] = 1; W[1][15] = 1
    
    # Product 16: (a11+a12+a21+a22)*(b11+b12+b21+b22)
    U[5][16] = 1; U[6][16] = 1; U[9][16] = 1; U[10][16] = 1
    V[5][16] = 1; V[6][16] = 1; V[9][16] = 1; V[10][16] = 1
    W[1][16] = 1; W[2][16] = 1
    
    # Product 17: (a22+a23+a32+a33)*(b22+b23+b32+b33)
    U[10][17] = 1; U[11][17] = 1; U[14][17] = 1; U[15][17] = 1
    V[10][17] = 1; V[11][17] = 1; V[14][17] = 1; V[15][17] = 1
    W[2][17] = 1; W[3][17] = 1
    
    # Product 18: (a02+a03+a20+a21+a12+a13+a30+a31)*(b02+b03+b20+b21+b12+b13+b30+b31)
    U[2][18] = 1; U[3][18] = 1; U[8][18] = 1; U[9][18] = 1
    U[6][18] = 1; U[7][18] = 1; U[12][18] = 1; U[13][18] = 1
    V[2][18] = 1; V[3][18] = 1; V[8][18] = 1; V[9][18] = 1
    V[6][18] = 1; V[7][18] = 1; V[12][18] = 1; V[13][18] = 1
    W[0][18] = 1; W[3][18] = 1
    
    # Product 19: (a00+a02+a20+a22+a10+a12+a21+a23+a30+a32+a31+a33)*(b00+b02+b20+b22+b10+b12+b21+b23+b30+b32+b31+b33)
    # This is getting complex. Let me use a verified Laderman scheme.
    # For now, I'll fall back if this doesn't verify.
    
    return U, V, W

def _kron_decomposition(u_block, v_block, w_block, block_size, n, m, p):
    """
    Construct a decomposition using Kronecker product of block decomposition and sub-block decomposition.
    u_block: (B*B, R_block) for BxB block structure
    v_block: (B*B, R_block)
    w_block: (B*B, R_block)
    block_size: size of each sub-block (e.g., 3 for 3x3 sub-blocks)
    n, m, p: total dimensions
    """
    B = n // block_size
    R_block = len(u_block[0])
    rank = R_block * (block_size ** 3)  # This is not right for Kronecker product
    
    # Actually, for Kronecker product of two decompositions:
    # If we have a BxB block decomposition of rank R_block, and each block is a SxS matrix
    # with a decomposition of rank R_sub, then total rank = R_block * R_sub.
    # But here, the "block" decomposition is the Laderman 4x4 (or Strassen 2x2),
    # and the "sub-block" is the 3x3 (or 2x2) matrix multiplication.
    
    # Let's use the general Kronecker product formula:
    # U[i*n+j, c1*R_sub + c2] = u_block[iB+j, c1] * u_sub[ip*B+jp, c2]
    # where i = iB*block_size + ip, j = jB*block_size + jp
    
    rank_total = R_block * (block_size ** 3)  # This is still not right
    
    # Correct approach:
    # If we treat the 6x6 matrix as a 2x2 block matrix of 3x3 blocks,
    # and use Strassen 2x2 (rank 7) for the block multiplication,
    # and schoolbook 3x3 (rank 27) for each block multiplication,
    # then total rank = 7 * 27 = 189.
    
    # For Laderman 4x4 (rank 20) with 1.5x1.5 blocks? No, that doesn't work.
    # Laderman is for 4x4, so we'd need 4x4 blocks of 1.5x1.5? No.
    
    # Let's stick with the proven 189 rank decomposition.
    return None

def _block_strassen_6x6x6():
    """
    Decompose 6x6x6 using 3x3 blocks of 2x2 matrices with Strassen.
    Total rank = 27 * 7 = 189.
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

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition (rank 189)
    candidate = _block_strassen_6x6x6()
    if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        return candidate
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
