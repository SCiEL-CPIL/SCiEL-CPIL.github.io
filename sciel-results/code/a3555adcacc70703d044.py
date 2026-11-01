"""Matrix multiplication 2x2,2x7 exact tensor decomposition using Strassen for 2x2x2 blocks."""

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    """
    Strassen's 7-term decomposition for 2x2 x 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) where:
      U index: i*m+j (i=0..1, j=0..1)
      V index: j*p+k (j=0..1, k=0..1)
      W index: k*n+i (k=0..1, i=0..1)
    
    Strassen formulas:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B01-B11)
      M4 = A11*(B10-B00)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B01)
      M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # U rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W rows: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3  [k*n+i]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = -1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = -1; W[3][1] = 1
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _direct_2x2x1():
    """
    Direct decomposition for 2x2 x 2x1 -> 2x1.
    Rank = 2.
    U (4x2), V (2x2), W (2x2)
    U index: i*m+j, V index: j*p+k = j*2+0 = j, W index: k*n+i = 0*2+i = i
    """
    rank = 2
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(2)]
    
    # C0 = A00*B0 + A01*B1
    U[0][0] = 1  # A00
    V[0][0] = 1  # B0
    W[0][0] = 1  # C0
    
    U[1][1] = 1  # A01
    V[1][1] = 1  # B1
    W[0][1] = 1  # C0
    
    # C1 = A10*B0 + A11*B1
    U[2][0] = 1  # A10
    V[0][0] = 1  # B0 (already set)
    W[1][0] = 1  # C1
    
    U[3][1] = 1  # A11
    V[1][1] = 1  # B1 (already set)
    W[1][1] = 1  # C1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    
    # Block structure:
    # B is 2x7. Split into: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:7]
    # First three blocks: 2x2x2 (Strassen, rank 7 each)
    # Last block: 2x2x1 (direct, rank 2)
    
    # Total rank = 3*7 + 2 = 23
    rank = 23
    
    U = [[0]*rank for _ in range(n*m)]  # 4 x 23
    V = [[0]*rank for _ in range(m*p)]  # 14 x 23
    W = [[0]*rank for _ in range(p*n)]  # 14 x 23
    
    # Get Strassen factors
    sU, sV, sW = _strassen_2x2x2()
    dU, dV, dW = _direct_2x2x1()
    
    # Block 1: columns 0-1 of B (k=0,1)
    col_offset = 0
    for c in range(7):
        for r in range(4):
            U[r][col_offset + c] = sU[r][c]
        V[r * p + 0][col_offset + c] = sV[r][c]  # j=0, k=0
        V[r * p + 1][col_offset + c] = sV[r][c]  # j=0, k=1 -- wait, V index is j*p+k
        # Actually V row index is j*p + k where j is row of B, k is col of B
        # For block 1: k=0,1 so V rows are 0*7+0=0, 0*7+1=1, 1*7+0=7, 1*7+1=8
        # But sV has 4 rows corresponding to (j,k) = (0,0),(0,1),(1,0),(1,1)
        # So we need to map sV row r to V row j*p+k where (j,k) is the r-th pair
    
    # Let me redo this more carefully
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen V rows: row 0=(j=0,k=0), row 1=(j=0,k=1), row 2=(j=1,k=0), row 3=(j=1,k=1)
    # Strassen W rows: row 0=(k=0,i=0), row 1=(k=0,i=1), row 2=(k=1,i=0), row 3=(k=1,i=1)
    
    # Block 1: k=0,1
    # V rows for block 1: j*p+k for j in {0,1}, k in {0,1}
    # j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 7; j=1,k=1 -> 8
    # W rows for block 1: k*n+i for k in {0,1}, i in {0,1}
    # k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    
    v_rows_block1 = [0, 1, 7, 8]  # maps to sV rows 0,1,2,3
    w_rows_block1 = [0, 1, 2, 3]  # maps to sW rows 0,1,2,3
    
    for c in range(7):
        for r in range(4):
            U[r][c] = sU[r][c]
        for r in range(4):
            V[v_rows_block1[r]][c] = sV[r][c]
            W[w_rows_block1[r]][c] = sW[r][c]
    
    # Block 2: k=2,3
    col_offset = 7
    v_rows_block2 = [2, 3, 9, 10]  # j=0,k=2->2; j=0,k=3->3; j=1,k=2->9; j=1,k=3->10
    w_rows_block2 = [4, 5, 6, 7]  # k=2,i=0->4; k=2,i=1->5; k=3,i=0->6; k=3,i=1->7
    
    for c in range(7):
        for r in range(4):
            U[r][col_offset + c] = sU[r][c]
        for r in range(4):
            V[v_rows_block2[r]][col_offset + c] = sV[r][c]
            W[w_rows_block2[r]][col_offset + c] = sW[r][c]
    
    # Block 3: k=4,5
    col_offset = 14
    v_rows_block3 = [4, 5, 11, 12]  # j=0,k=4->4; j=0,k=5->5; j=1,k=4->11; j=1,k=5->12
    w_rows_block3 = [8, 9, 10, 11]  # k=4,i=0->8; k=4,i=1->9; k=5,i=0->10; k=5,i=1->11
    
    for c in range(7):
        for r in range(4):
            U[r][col_offset + c] = sU[r][c]
        for r in range(4):
            V[v_rows_block3[r]][col_offset + c] = sV[r][c]
            W[w_rows_block3[r]][col_offset + c] = sW[r][c]
    
    # Block 4: k=6
    col_offset = 21
    # dV has 2 rows: row 0=(j=0,k=0), row 1=(j=1,k=0)
    # dW has 2 rows: row 0=(k=0,i=0), row 1=(k=0,i=1)
    # For k=6: V rows: j=0,k=6->6; j=1,k=6->13
    # W rows: k=6,i=0->12; k=6,i=1->13
    
    v_rows_block4 = [6, 13]
    w_rows_block4 = [12, 13]
    
    for c in range(2):
        for r in range(4):
            U[r][col_offset + c] = dU[r][c]
        for r in range(2):
            V[v_rows_block4[r]][col_offset + c] = dV[r][c]
            W[w_rows_block4[r]][col_offset + c] = dW[r][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
