"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is correct using vectorized NumPy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k])  # This is wrong, need element-wise then dot with W
                # Actually: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # M1 = (a11+a22)(b11+b22) -> contributes to c11, c22
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # M2 = (a21+a22)b11 -> contributes to c21
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0
    
    # M3 = a11(b12-b22) -> contributes to c12
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0
    
    # M4 = a22(b21-b11) -> contributes to c11, c21
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[2][3] = 1.0
    
    # M5 = (a11+a12)b22 -> contributes to c11, c12
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[1][4] = 1.0
    
    # M6 = (a21-a11)(b11+b12) -> contributes to c22
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # M7 = (a12-a22)(b21+b22) -> contributes to c11, c22
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0; W[3][6] = 1.0
    
    return U, V, W

def _build_2x2x1():
    """Return rank-2 schoolbook decomposition for 2x2x1 matrix multiplication."""
    n, m, p = 2, 2, 1
    rank = n * m * p  # = 4, but we can do better? No, it's already minimal
    # Actually for 2x2x1: C[i,0] = sum_j A[i,j]*B[j,0]
    # Need: A[i,0]*B[0,0] + A[i,1]*B[1,0] for i=0,1
    # This requires 4 scalar products, so rank 4? 
    # Wait: the tensor has 2*2*1=8 entries, but many are zero.
    # T[i,j,0] = 1 for all i,j. So we need sum_r U[i,j,r]*V[j,0,r]*W[0,i,r] = 1
    # Minimum rank: each (i,j) pair needs its own term since W[0,i,r] couples i.
    # Actually we can share: let r=0 handle (i=0,j=0), r=1 handle (i=0,j=1), etc.
    # But W[0,0,r] and W[0,1,r] are different, so we need separate terms.
    # Rank = 4 for schoolbook.
    rank = 4
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1.0
            V[j*p+0][col] = 1.0
            W[0*n+i][col] = 1.0
            col += 1
    
    return U, V, W

def _build_2x2x3():
    """Return rank-8 decomposition for 2x2x3 matrix multiplication."""
    # Split 3 columns into 2+1
    # M_{2,2,2} rank 7 + M_{2,2,1} rank 4 = 11? That's worse than schoolbook 12.
    # Actually schoolbook for 2x2x3 is 12. Let me just use schoolbook.
    # Or: use Strassen for first 2 cols (rank 7) + schoolbook for last col (rank 4) = 11
    # But 11 > 12? No, 11 < 12. So rank 11 is better.
    # Wait, I need to check if this is valid.
    
    n, m, p = 2, 2, 3
    # Use Strassen for cols 0,1 and schoolbook for col 2
    U_s, V_s, W_s = _build_strassen_2x2x2()
    U_b, V_b, W_b = _build_2x2x1()
    
    rank = 7 + 4  # = 11
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # Copy Strassen part (cols 0,1)
    for i in range(n*m):
        for r in range(7):
            U[i][r] = U_s[i][r]
    for j in range(2):  # m=2, only first 2 cols of V
        for r in range(7):
            V[j*p+0][r] = V_s[j*p+0][r]
            V[j*p+1][r] = V_s[j*p+1][r]
    for k in range(2):  # only first 2 cols of W
        for i in range(n):
            W[k*n+i][r if False else r] = 0  # fix
    # Redo W properly
    W = [[0.0] * rank for _ in range(p*n)]
    for k in range(2):
        for i in range(n):
            for r in range(7):
                W[k*n+i][r] = W_s[k*n+i][r]
    
    # Copy schoolbook part for col 2, offset by 7
    for i in range(n*m):
        for r in range(4):
            U[i][7+r] = U_b[i][r]
    for j in range(m):
        for r in range(4):
            V[j*p+2][7+r] = V_b[j*p+0][r]
    for i in range(n):
        for r in range(4):
            W[2*n+i][7+r] = W_b[0*n+i][r]
    
    return U, V, W

def _build_2x2x4():
    """Return rank-14 decomposition for 2x2x4 using two Strassen blocks."""
    n, m, p = 2, 2, 4
    U1, V1, W1 = _build_strassen_2x2x2()
    U2, V2, W2 = _build_strassen_2x2x2()
    
    rank = 14
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # First Strassen for cols 0,1
    for i in range(n*m):
        for r in range(7):
            U[i][r] = U1[i][r]
    for j in range(m):
        for r in range(7):
            V[j*p+0][r] = V1[j*p+0][r]
            V[j*p+1][r] = V1[j*p+1][r]
    for k in range(2):
        for i in range(n):
            for r in range(7):
                W[k*n+i][r] = W1[k*n+i][r]
    
    # Second Strassen for cols 2,3
    for i in range(n*m):
        for r in range(7):
            U[i][7+r] = U2[i][r]
    for j in range(m):
        for r in range(7):
            V[j*p+2][7+r] = V2[j*p+0][r]
            V[j*p+3][7+r] = V2[j*p+1][r]
    for k in range(2, 4):
        for i in range(n):
            for r in range(7):
                W[k*n+i][7+r] = W2[(k-2)*n+i][r]
    
    return U, V, W

def _build_2x2x7():
    """Return decomposition for 2x2x7 using Strassen blocks."""
    n, m, p = 2, 2, 7
    # Split 7 cols into 4+3
    U4, V4, W4 = _build_2x2x4()   # rank 14
    U3, V3, W3 = _build_2x2x3()   # rank 11
    
    rank = 14 + 11  # = 25
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # Copy 2x2x4 part (cols 0-3)
    for i in range(n*m):
        for r in range(14):
            U[i][r] = U4[i][r]
    for j in range(m):
        for c in range(4):
            for r in range(14):
                V[j*p+c][r] = V4[j*p+c][r]
    for k in range(4):
        for i in range(n):
            for r in range(14):
                W[k*n+i][r] = W4[k*n+i][r]
    
    # Copy 2x2x3 part (cols 4-6), offset by 14
    for i in range(n*m):
        for r in range(11):
            U[i][14+r] = U3[i][r]
    for j in range(m):
        for c in range(3):
            for r in range(11):
                V[j*p+(4+c)][14+r] = V3[j*p+c][r]
    for k in range(3):
        for i in range(n):
            for r in range(11):
                W[(4+k)*n+i][14+r] = W3[k*n+i][r]
    
    return U, V, W

def _build_2x4x7_decomposition():
    """Build a decomposition for 2x4x7 using block Strassen methods."""
    n, m, p = 2, 4, 7
    
    # Split A (2x4) into A1 (2x2) cols 0-1, A2 (2x2) cols 2-3
    # Split B (4x7) into B1 (2x7) rows 0-1, B2 (2x7) rows 2-3
    # A@B = A1@B1 + A2@B2
    # Each is M_{2,2,7}
    
    U1, V1, W1 = _build_2x2x7()  # rank 25
    U2, V2, W2 = _build_2x2x7()  # rank 25
    
    rank = 25 + 25  # = 50
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # A1@B1: A1 is rows of A with cols 0-1, B1 is rows 0-1 of B
    # U indices: i*m+j where i in {0,1}, j in {0,1} for A1 part
    # V indices: j*p+k where j in {0,1}, k in {0..6} for B1 part
    # W indices: k*n+i
    
    for i in range(n):
        for j in range(2):  # cols 0-1 of A
            for r in range(25):
                U[i*m+j][r] = U1[i*2+j][r]
    
    for j in range(2):  # rows 0-1 of B
        for k in range(p):
            for r in range(25):
                V[j*p+k][r] = V1[j*2+k][r]
    
    for k in range(p):
        for i in range(n):
            for r in range(25):
                W[k*n+i][r] = W1[k*n+i][r]
    
    # A2@B2: A2 is cols 2-3, B2 is rows 2-3
    offset = 25
    for i in range(n):
        for j in range(2, 4):  # cols 2-3 of A
            for r in range(25):
                U[i*m+j][offset+r] = U2[i*2+(j-2)][r]
    
    for j in range(2, 4):  # rows 2-3 of B
        for k in range(p):
            for r in range(25):
                V[j*p+k][offset+r] = V2[(j-2)*2+k][r]
    
    for k in range(p):
        for i in range(n):
            for r in range(25):
                W[k*n+i][offset+r] = W2[k*n+i][r]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Try the Strassen-based decomposition (rank 50)
    U, V, W = _build_2x4x7_decomposition()
    rank = len(U[0])
    
    # Verify
    if _verify_decomposition(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (rank 56)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
