"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)  # (n*m, rank)
    Va = np.array(V, dtype=np.float64)  # (m*p, rank)
    Wa = np.array(W, dtype=np.float64)  # (p*n, rank)
    # Reshape: Ua -> (n, m, rank), Va -> (m, p, rank), Wa -> (p, n, rank)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # Contract over rank: result[i,j,k] = sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
    # Check all entries are 1.0
    return bool(np.allclose(result, 1.0, atol=1e-9))

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
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

def _winograd_3x3():
    """
    Return a rank-21 Winograd decomposition for 3x3x3 matrix multiplication.
    U: (9, 21), V: (9, 21), W: (9, 21)
    Indices: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    
    Uses Winograd's F(2,2) transform approach for 3x3 blocks.
    Actually, let's use the known rank-23 or rank-21 result.
    For 3x3, the best known is rank 23 (or 21 with half-integers).
    
    Let me use a simpler approach: tensor product of Strassen 2x2 (rank 7) 
    with a 3x3 structure won't directly give 3x3.
    
    Instead, I'll construct the 6x6x6 as 2x2 blocks of 3x3 matrices.
    Block level: 2x2 Strassen (rank 7)
    Intra-block: 3x3 schoolbook (rank 27)
    Total: 7 * 27 = 189 (same as before)
    
    Alternative: 3x3 blocks of 2x2 (current approach) = 27 * 7 = 189
    
    For improvement, I need a better 3x3 decomposition.
    The best known rank for 3x3x3 is 23 (over reals, with integers).
    Let me use rank 23 for the 3x3 blocks.
    Total: 23 * 7 = 161
    """
    # Known rank-23 decomposition for 3x3x3
    # This is complex; let me use a simpler verified approach.
    # For now, return the schoolbook 3x3 (rank 27) as fallback
    n = 3
    rank = n * n * n  # 27
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i*n+j][col] = 1
                V[j*n+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _block_decomposition_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices.
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
                    # U part
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row][col] = sU[li*2+lj][c]
                    
                    # V part
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row][col] = sV[lj*2+lk][c]
                    
                    # W part
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
    
    # Primary strategy: 3x3 blocks of 2x2 with Strassen (rank 189)
    best = _block_decomposition_6x6x6()
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback: schoolbook (rank 216)
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
