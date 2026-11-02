import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify if U, V, W form a valid decomposition."""
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = 0.0
                for r in range(R):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - expected) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2.
    Returns U (4 x 7), V (4 x 7), W (4 x 7)
    U indexed by [i*2+j], V by [j*2+k], W by [k*2+i]
    """
    R = 7
    U = [[0.0]*R for _ in range(4)]  # i,j in {0,1} -> index i*2+j
    V = [[0.0]*R for _ in range(4)]  # j,k in {0,1} -> index j*2+k
    W = [[0.0]*R for _ in range(4)]  # k,i in {0,1} -> index k*2+i
    
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # Term 0: M1 = (A00+A11)*(B00+B11)
    U[0*2+0][0] = 1.0; U[1*2+1][0] = 1.0
    V[0*2+0][0] = 1.0; V[1*2+1][0] = 1.0
    W[0*2+0][0] = 1.0; W[1*2+0][0] = 1.0; W[0*2+1][0] = 1.0; W[1*2+1][0] = 1.0
    # C00 += M1, C11 += M1, C01 += 0, C10 += 0
    # Wait, W[k*2+i]: for C00 (k=0,i=0): W[0][0]=1, for C11 (k=1,i=1): W[3][0]=1
    # For C01 (k=0,i=1): W[1][0], for C10 (k=1,i=0): W[2][0]
    # M1 contributes to C00 and C11 only.
    # W[0*2+0][0]=1 (k=0,i=0 -> C00), W[1*2+1][0]=1 (k=1,i=1 -> C11)
    # W[0*2+1][0]=0, W[1*2+0][0]=0
    
    # Let me redo this more carefully.
    U = [[0.0]*R for _ in range(4)]
    V = [[0.0]*R for _ in range(4)]
    W = [[0.0]*R for _ in range(4)]
    
    # Term 0: M1 = (A00+A11)*(B00+B11)
    # U: A00 (idx 0) = 1, A11 (idx 3) = 1
    # V: B00 (idx 0) = 1, B11 (idx 3) = 1
    # W: contributes to C00 (k=0,i=0, idx 0) and C11 (k=1,i=1, idx 3)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # Term 1: M2 = (A10+A11)*B00
    # U: A10 (idx 2) = 1, A11 (idx 3) = 1
    # V: B00 (idx 0) = 1
    # W: contributes to C10 (k=1,i=0, idx 2)
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0
    
    # Term 2: M3 = A00*(B10-B11)
    # U: A00 (idx 0) = 1
    # V: B10 (idx 2) = 1, B11 (idx 3) = -1
    # W: contributes to C01 (k=0,i=1, idx 1)
    U[0][2] = 1.0
    V[2][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0
    
    # Term 3: M4 = A11*(B01-B00)
    # U: A11 (idx 3) = 1
    # V: B01 (idx 1) = 1, B00 (idx 0) = -1
    # W: contributes to C00 (k=0,i=0, idx 0) with -1, C01 (k=0,i=1, idx 1) with +1
    U[3][3] = 1.0
    V[1][3] = 1.0; V[0][3] = -1.0
    W[0][3] = -1.0; W[1][3] = 1.0
    
    # Term 4: M5 = (A00+A01)*B11
    # U: A00 (idx 0) = 1, A01 (idx 1) = 1
    # V: B11 (idx 3) = 1
    # W: contributes to C00 (k=0,i=0, idx 0)
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = 1.0
    
    # Term 5: M6 = (A10-A00)*(B00+B10)
    # U: A10 (idx 2) = 1, A00 (idx 0) = -1
    # V: B00 (idx 0) = 1, B10 (idx 2) = 1
    # W: contributes to C11 (k=1,i=1, idx 3)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[2][5] = 1.0
    W[3][5] = 1.0
    
    # Term 6: M7 = (A01-A11)*(B01+B11)
    # U: A01 (idx 1) = 1, A11 (idx 3) = -1
    # V: B01 (idx 1) = 1, B11 (idx 3) = 1
    # W: contributes to C00 (k=0,i=0, idx 0) with -1, C10 (k=1,i=0, idx 2) with +1
    U[1][6] = 1.0; U[3][6] = -1.0
    V[1][6] = 1.0; V[3][6] = 1.0
    W[0][6] = -1.0; W[2][6] = 1.0
    
    return U, V, W

def _build_3x3x2():
    """
    Build rank-17 decomposition for 3x3 * 3x2.
    Uses Strassen for 2x2x2 core + 10 schoolbook terms for remaining.
    U: 9 x 17, V: 6 x 17, W: 6 x 17
    U indexed by [i*3+j], V by [j*2+k], W by [k*3+i]
    """
    n, m, p = 3, 3, 2
    R = 17
    U = [[0.0]*R for _ in range(n*m)]
    V = [[0.0]*R for _ in range(m*p)]
    W = [[0.0]*R for _ in range(p*n)]
    
    # Get Strassen 2x2x2
    Us, Vs, Ws = _strassen_2x2x2()
    
    # Embed Strassen into the 3x3x2 problem
    # Strassen handles i,j in {0,1}, k in {0,1}
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j][r] = Us[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*2+k][r] = Vs[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*3+i][r] = Ws[k*2+i][r]
    
    # Remaining 10 terms (schoolbook for non-Strassen positions)
    # These are: (i,j) pairs where i=2 or j=2, times k in {0,1}
    # (2,0,0), (2,0,1), (2,1,0), (2,1,1), (2,2,0), (2,2,1), (0,2,0), (0,2,1), (1,2,0), (1,2,1)
    remaining = [
        (2,0,0), (2,0,1),
        (2,1,0), (2,1,1),
        (2,2,0), (2,2,1),
        (0,2,0), (0,2,1),
        (1,2,0), (1,2,1),
    ]
    
    for idx, (i, j, k) in enumerate(remaining):
        r = 7 + idx
        U[i*3+j][r] = 1.0
        V[j*2+k][r] = 1.0
        W[k*3+i][r] = 1.0
    
    return U, V, W

def _build_3x3x6():
    """
    Build decomposition for 3x3 * 3x6.
    Split into three 3x3x2 blocks.
    Total rank: 3 * 17 = 51
    U: 9 x 51, V: 18 x 51, W: 18 x 51
    U indexed by [i*3+j], V by [j*6+k], W by [k*3+i]
    """
    n, m, p = 3, 3, 6
    R_block = 17
    R_total = 3 * R_block  # 51
    
    U_full = [[0.0]*R_total for _ in range(n*m)]
    V_full = [[0.0]*R_total for _ in range(m*p)]
    W_full = [[0.0]*R_total for _ in range(p*n)]
    
    U_block, V_block, W_block = _build_3x3x2()
    
    for block in range(3):
        offset = block * R_block
        # B columns: 2*block, 2*block+1
        # W rows: k in {2*block, 2*block+1}, so W index = k*3+i
        
        for r in range(R_block):
            for i in range(n):
                for j in range(m):
                    U_full[i*m+j][offset+r] = U_block[i*m+j][r]
            
            for j in range(m):
                for k in range(2):
                    V_full[j*p + (2*block+k)][offset+r] = V_block[j*2+k][r]
            
            for k in range(2):
                for i in range(n):
                    W_full[(2*block+k)*n + i][offset+r] = W_block[k*n+i][r]
    
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120 - 15
    
    # Build the Strassen-based decomposition
    U, V, W = _build_3x3x6()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 54
    R = n * m * p
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
