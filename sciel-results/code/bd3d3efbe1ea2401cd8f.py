"""
Program for Matrix Multiplication 2x3x5 Tensor Decomposition.
Strategy: Block decomposition using Strassen's algorithm for 2x2x2 sub-problems.
Target Rank: 28 (vs 30 for schoolbook).
"""

import time

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices:
    U: (i, j) for A[i, j]
    V: (j, k) for B[j, k]
    W: (k, i) for C[k, i] -- Wait, trace convention W[k*n+i, r]
    For 2x2x2: n=2, m=2, p=2.
    U index: i*m + j = i*2 + j
    V index: j*p + k = j*2 + k
    W index: k*n + i = k*2 + i
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W entries.
    # Each term r corresponds to a rank-1 tensor u_r \otimes v_r \otimes w_r.
    # U[i*m+j, r] is the coefficient for A[i,j] in term r.
    # V[j*p+k, r] is the coefficient for B[j,k] in term r.
    # W[k*n+i, r] is the coefficient for C[k,i] in term r.
    
    # Term 1: M1
    # (A11 + A22)(B11 + B22)
    # A part: A[0,0]=1, A[1,1]=1
    # B part: B[0,0]=1, B[1,1]=1
    # C part: C[0,0] += 1, C[1,1] += 1 (from M1 in C11 and C22)
    # Wait, the expansion is:
    # C11 gets +M1
    # C22 gets +M1
    # So W[0*2+0, 0] = 1 (C[0,0])
    # W[1*2+1, 0] = 1 (C[1,1])
    # U[0*2+0, 0] = 1 (A[0,0])
    # U[1*2+1, 0] = 1 (A[1,1])
    # V[0*2+0, 0] = 1 (B[0,0])
    # V[1*2+1, 0] = 1 (B[1,1])
    
    # This is getting tedious to do by hand for all 7 terms.
    # I will use a known implementation of Strassen in tensor form.
    
    # Let's define the 7 vectors explicitly.
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    U = [
        [1, 0, 0, 1, 1, 0, 1], # A11, A12, A21, A22 for each term
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1]
    ]
    V = [
        [1, 0, 0, 1, 0, 1, 0], # B11, B12, B21, B22
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 1, 0, 1],
        [0, 0, 1, -1, 1, 0, 1]
    ]
    # Wait, Strassen coefficients are usually integers.
    # Let's re-verify the standard Strassen matrix.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's build U, V, W based on this.
    # Term 1 (M1):
    # U: A11=1, A22=1 -> [1, 0, 0, 1]
    # V: B11=1, B22=1 -> [1, 0, 0, 1]
    # W: C11=1, C22=1 -> [1, 0, 0, 1] (indices: 00, 01, 10, 11)
    
    # Term 2 (M2):
    # U: A21=1, A22=1 -> [0, 0, 1, 1]
    # V: B11=1 -> [1, 0, 0, 0]
    # W: C21=1 -> [0, 0, 1, 0]
    
    # Term 3 (M3):
    # U: A11=1 -> [1, 0, 0, 0]
    # V: B21=1, B22=-1 -> [0, 0, 1, -1]
    # W: C12=1, C22=1 -> [0, 1, 0, 1]
    
    # Term 4 (M4):
    # U: A22=1 -> [0, 0, 0, 1]
    # V: B12=1, B11=-1 -> [-1, 1, 0, 0]
    # W: C11=1, C21=1 -> [1, 0, 1, 0]
    
    # Term 5 (M5):
    # U: A11=1, A12=1 -> [1, 1, 0, 0]
    # V: B22=1 -> [0, 0, 0, 1]
    # W: C11=-1, C12=1 -> [-1, 1, 0, 0]
    
    # Term 6 (M6):
    # U: A21=1, A11=-1 -> [-1, 0, 1, 0]
    # V: B11=1, B12=1 -> [1, 1, 0, 0]
    # W: C22=1 -> [0, 0, 0, 1]
    
    # Term 7 (M7):
    # U: A12=1, A22=-1 -> [0, 1, 0, -1]
    # V: B21=1, B22=1 -> [0, 0, 1, 1]
    # W: C11=1 -> [1, 0, 0, 0]
    
    # Now map to flat indices.
    # A: 00=0, 01=1, 10=2, 11=3
    # B: 00=0, 01=1, 10=2, 11=3
    # C: 00=0, 01=1, 10=2, 11=3
    
    U_flat = [
        [1, 0, 0, 1, 1, -1, 0], # Row 0 (A11)
        [0, 0, 0, 0, 1, 0, 1],  # Row 1 (A12)
        [0, 0, 0, 0, 0, 1, 0],  # Row 2 (A21)
        [1, 1, 0, 1, 0, 0, -1]  # Row 3 (A22)
    ]
    V_flat = [
        [1, 1, 0, -1, 0, 1, 0], # Row 0 (B11)
        [0, 0, 0, 1, 0, 1, 0],  # Row 1 (B12)
        [0, 0, 1, 0, 0, 0, 1],  # Row 2 (B21)
        [1, 0, -1, 0, 1, 0, 1]  # Row 3 (B22)
    ]
    W_flat = [
        [1, 0, 0, 1, -1, 0, 1], # Row 0 (C11)
        [0, 0, 1, 0, 1, 0, 0],  # Row 1 (C12)
        [0, 1, 0, 1, 0, 0, 0],  # Row 2 (C21)
        [1, 0, 1, 0, 0, 1, 1]   # Row 3 (C22)
    ]
    
    return U_flat, V_flat, W_flat

def _schoolbook_2x1x2():
    """
    Returns U, V, W for 2x1x2 matrix multiplication with rank 4.
    A is 2x1, B is 1x2, C is 2x2.
    U shape: (2, 4), V shape: (2, 4), W shape: (4, 4)
    """
    n, m, p = 2, 1, 2
    rank = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _schoolbook_2x3x1():
    """
    Returns U, V, W for 2x3x1 matrix multiplication with rank 6.
    A is 2x3, B is 3x1, C is 2x1.
    U shape: (6, 6), V shape: (3, 6), W shape: (2, 6)
    """
    n, m, p = 2, 3, 1
    rank = 6
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _construct_2x3x2(U_strassen, V_strassen, W_strassen, U_sb, V_sb, W_sb):
    """
    Combines Strassen 2x2x2 and Schoolbook 2x1x2 to form 2x3x2.
    A is 2x3, B is 3x2, C is 2x2.
    Partition A: A1 (2x2), A2 (2x1).
    Partition B: B1 (2x2), B2 (1x2).
    C = A1 B1 + A2 B2.
    
    U total shape: (6, 11)
    V total shape: (6, 11)
    W total shape: (4, 11)
    """
    n, m, p = 2, 3, 2
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A1 (2x2) * B1 (2x2) -> C1 (2x2)
    # A1 indices in A (2x3): (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # B1 indices in B (3x2): (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # C1 indices in C (2x2): 0, 1, 2, 3
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i * 3 + j][r] = U_strassen[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * 2 + k][r] = V_strassen[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][r] = W_strassen[k * 2 + i][r]
        
    # Part 2: A2 (2x1) * B2 (1x2) -> C2 (2x2)
    # A2 indices in A (2x3): (0,2), (1,2) -> 2, 5
    # B2 indices in B (3x2): (2,0), (2,1) -> 4, 5
    # C2 indices in C (2x2): 0, 1, 2, 3
    
    for r in range(4):
        r_off = 7 + r
        for i in range(2):
            U[i * 3 + 2][r_off] = U_sb[i * 1 + 0][r] # A[i, 0] in 2x1 is at i*1+0
        for k in range(2):
            V[2 * 2 + k][r_off] = V_sb[0 * 2 + k][r] # B[0, k] in 1x2 is at 0*2+k
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][r_off] = W_sb[k * 2 + i][r]
                
    return U, V, W

def construct():
    """
    Returns the block decomposition for 2x3x5 matrix multiplication.
    Rank = 28.
    """
    start_time = time.time()
    deadline = start_time + 120 # Safety margin
    
    # Get base decompositions
    U_s, V_s, W_s = _strassen_2x2x2()
    U_sb_212, V_sb_212, W_sb_212 = _schoolbook_2x1x2()
    U_sb_231, V_sb_231, W_sb_231 = _schoolbook_2x3x1()
    
    # Construct 2x3x2 decomposition (rank 11)
    U_232, V_232, W_232 = _construct_2x3x2(U_s, V_s, W_s, U_sb_212, V_sb_212, W_sb_212)
    
    # Final 2x3x5 decomposition
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: Columns 0-1 of B (2x3x2)
    # A: 2x3 (indices 0-5)
    # B: 3x2 (indices 0-5) -> B[0:2, 0:2] and B[2:3, 0:2]
    # C: 2x2 (indices 0-3) -> C[0:2, 0:2]
    
    for r in range(11):
        r_off = r
        for i in range(2):
            for j in range(3):
                U[i * 3 + j][r_off] = U_232[i * 3 + j][r]
        for j in range(3):
            for k in range(2):
                V[j * 5 + k][r_off] = V_232[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][r_off] = W_232[k * 2 + i][r]
                
    # Block 2: Columns 2-3 of B (2x3x2)
    # A: 2x3 (indices 0-5)
    # B: 3x2 (indices 10-15) -> B[0:2, 2:4] and B[2:3, 2:4]
    # C: 2x2 (indices 4-7) -> C[0:2, 2:4]
    
    for r in range(11):
        r_off = 11 + r
        for i in range(2):
            for j in range(3):
                U[i * 3 + j][r_off] = U_232[i * 3 + j][r]
        for j in range(3):
            for k in range(2):
                V[j * 5 + (k + 2)][r_off] = V_232[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * 2 + i][r_off] = W_232[k * 2 + i][r]
                
    # Block 3: Column 4 of B (2x3x1)
    # A: 2x3 (indices 0-5)
    # B: 3x1 (indices 20) -> B[0:3, 4]
    # C: 2x1 (indices 8-9) -> C[0:2, 4]
    
    for r in range(6):
        r_off = 22 + r
        for i in range(2):
            for j in range(3):
                U[i * 3 + j][r_off] = U_sb_231[i * 3 + j][r]
        for j in range(3):
            V[j * 5 + 4][r_off] = V_sb_231[j * 1 + 0][r]
        for k in range(1):
            for i in range(2):
                W[4 * 2 + i][r_off] = W_sb_231[0 * 2 + i][r]
                
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
