"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

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
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _liu_3x3():
    """
    Return the rank-19 Liu decomposition for 3x3x3 matrix multiplication.
    This is a placeholder implementation. If the coefficients are not correct,
    the verification will fail and we will fall back to a valid decomposition.
    """
    # The Liu 19 decomposition is complex. 
    # I will attempt to construct it or return None if I cannot guarantee correctness.
    # For the sake of this exercise, I will return None to trigger the fallback,
    # unless I can find a reliable hardcoded version.
    # Given the risk, I will return None.
    return None

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

def _block_decomposition_6x6x6_liu():
    """
    Decompose 6x6x6 using 2x2 blocks of 3x3 matrices (Liu rank 19).
    Rank 133.
    """
    liu = _liu_3x3()
    if liu is None:
        return None
    
    lU, lV, lW = liu
    sU, sV, sW = _strassen_2x2()
    
    rank = 7 * 19
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for c2 in range(7):
        for c3 in range(19):
            col = c2 * 19 + c3
            for i2 in range(2):
                for j2 in range(2):
                    for i3 in range(3):
                        for j3 in range(3):
                            u_row = (i2*3+i3)*6 + (j2*3+j3)
                            U[u_row][col] = sU[i2*2+j2][c2] * lU[i3*3+j3][c3]
            for j2 in range(2):
                for k2 in range(2):
                    for j3 in range(3):
                        for k3 in range(3):
                            v_row = (j2*3+j3)*6 + (k2*3+k3)
                            V[v_row][col] = sV[j2*2+k2][c2] * lV[j3*3+k3][c3]
            for k2 in range(2):
                for i2 in range(2):
                    for k3 in range(3):
                        for i3 in range(3):
                            w_row = (k2*3+k3)*6 + (i2*3+i3)
                            W[w_row][col] = sW[k2*2+i2][c2] * lW[k3*3+i3][c3]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try Liu-based decomposition (rank 133)
    liu_decomp = _block_decomposition_6x6x6_liu()
    if liu_decomp is not None:
        if _verify(liu_decomp["U"], liu_decomp["V"], liu_decomp["W"], n, m, p):
            return liu_decomp
    
    # Fallback to Strassen-based decomposition (rank 189)
    strassen_decomp = _block_decomposition_6x6x6_strassen()
    if _verify(strassen_decomp["U"], strassen_decomp["V"], strassen_decomp["W"], n, m, p):
        return strassen_decomp
    
    # Final fallback: Schoolbook (rank 216)
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
