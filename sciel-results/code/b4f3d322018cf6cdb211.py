"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2 * 2x2 multiplication using Strassen's algorithm (rank 7).
    
    A = [[a, b], [c, d]]  (n=2, m=2)
    B = [[e, f], [g, h]]  (m=2, p=2)
    C = A * B = [[c11, c12], [c21, c22]]
    
    Strassen products:
    P1 = (a+d)(e+h)
    P2 = (c+d)(e)
    P3 = (a)(f-h)
    P4 = (d)(g-e)
    P5 = (a+b)(h)
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    U has shape (n*m, rank) = (4, 7): rows indexed by [a,b,c,d] -> i*m+j
    V has shape (m*p, rank) = (4, 7): rows indexed by [e,f,g,h] -> j*p+k
    W has shape (p*n, rank) = (4, 7): rows indexed by C[k*n+i]
    
    For C = A*B:
    C[0][0] = c11 -> index k*n+i = 0*2+0 = 0
    C[0][1] = c12 -> index k*n+i = 1*2+0 = 2
    C[1][0] = c21 -> index k*n+i = 0*2+1 = 1
    C[1][1] = c22 -> index k*n+i = 1*2+1 = 3
    """
    rank = 7
    # U rows: a=0, b=1, c=2, d=3
    # V rows: e=0, f=1, g=2, h=3
    # W rows: c11=0, c21=1, c12=2, c22=3
    
    # P1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: c11=+1, c22=+1
    # P2 = (c+d)(e):     U: c=1,d=1; V: e=1;      W: c21=+1, c22=-1
    # P3 = (a)(f-h):     U: a=1;        V: f=1,h=-1; W: c12=+1, c22=+1
    # P4 = (d)(g-e):     U: d=1;        V: g=1,e=-1; W: c11=+1, c21=+1
    # P5 = (a+b)(h):     U: a=1,b=1; V: h=1;      W: c11=-1, c12=+1
    # P6 = (c-a)(e+f):   U: c=1,a=-1; V: e=1,f=1; W: c22=+1
    # P7 = (b-d)(g+h):   U: b=1,d=-1; V: g=1,h=1; W: c11=+1
    
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # P1: col 0
    U[0][0] = 1; U[3][0] = 1  # a, d
    V[0][0] = 1; V[3][0] = 1  # e, h
    W[0][0] = 1; W[3][0] = 1  # c11, c22
    
    # P2: col 1
    U[2][1] = 1; U[3][1] = 1  # c, d
    V[0][1] = 1              # e
    W[1][1] = 1; W[3][1] = -1  # c21, c22
    
    # P3: col 2
    U[0][2] = 1              # a
    V[1][2] = 1; V[3][2] = -1  # f, -h
    W[2][2] = 1; W[3][2] = 1  # c12, c22
    
    # P4: col 3
    U[3][3] = 1              # d
    V[2][3] = 1; V[0][3] = -1  # g, -e
    W[0][3] = 1; W[1][3] = 1  # c11, c21
    
    # P5: col 4
    U[0][4] = 1; U[1][4] = 1  # a, b
    V[3][4] = 1              # h
    W[0][4] = -1; W[2][4] = 1  # -c11, c12
    
    # P6: col 5
    U[2][5] = 1; U[0][5] = -1  # c, -a
    V[0][5] = 1; V[1][5] = 1  # e, f
    W[3][5] = 1              # c22
    
    # P7: col 6
    U[1][6] = 1; U[3][6] = -1  # b, -d
    V[2][6] = 1; V[3][6] = 1  # g, h
    W[0][6] = 1              # c11
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Return U, V, W for 2x2 * 2x1 multiplication (schoolbook, rank 4).
    
    A = [[a, b], [c, d]] (n=2, m=2)
    B = [[e], [f]] (m=2, p=1)
    C = A*B = [[a*e + b*f], [c*e + d*f]]
    
    C[0] = a*e + b*f -> index k*n+i = 0*2+0 = 0
    C[1] = c*e + d*f -> index k*n+i = 0*2+1 = 1
    
    U rows: a=0, b=1, c=2, d=3
    V rows: e=0, f=1 (j*p+k, p=1: j=0->k=0, j=1->k=1)
    W rows: C0=0, C1=1
    """
    rank = 4
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(2)]
    
    # a*e: col 0
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # b*f: col 1
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # c*e: col 2
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # d*f: col 3
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W

def construct():
    """
    Block decomposition: split 13 columns into 6 pairs + 1 single.
    Each pair uses Strassen (rank 7), single uses schoolbook (rank 4).
    Total rank = 6*7 + 4 = 46.
    """
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Get base decompositions
    sU, sV, sW = _strassen_2x2x2()
    bU, bV, bW = _schoolbook_2x2x1()
    
    total_rank = 6 * 7 + 4  # 46
    
    U = [[0] * total_rank for _ in range(n * m)]      # (4, 46)
    V = [[0] * total_rank for _ in range(m * p)]      # (26, 46)
    W = [[0] * total_rank for _ in range(p * n)]      # (26, 46)
    
    col_offset = 0
    
    # Process 6 pairs of columns
    for block in range(6):
        k1 = 2 * block
        k2 = 2 * block + 1
        
        # Strassen block: 7 columns
        for c in range(7):
            col = col_offset + c
            
            # U: same for all blocks (4 entries)
            for i in range(4):
                U[i][col] = sU[i][c]
            
            # V: rows j*p+k for j=0,1 and k=k1,k2
            # V row index = j*p + k
            # sV row 0 -> e (j=0, k=k1) -> V[0*13+k1]
            # sV row 1 -> f (j=0, k=k2) -> V[0*13+k2]
            # sV row 2 -> g (j=1, k=k1) -> V[1*13+k1]
            # sV row 3 -> h (j=1, k=k2) -> V[1*13+k2]
            v_mapping = [k1, k2, 13 + k1, 13 + k2]
            for i in range(4):
                V[v_mapping[i]][col] = sV[i][c]
            
            # W: rows k*n+i for k=0,1 and i=0,1
            # sW row 0 -> c11 (k=0, i=0) -> W[0*2+0] = W[0]
            # sW row 1 -> c21 (k=0, i=1) -> W[0*2+1] = W[1]
            # sW row 2 -> c12 (k=1, i=0) -> W[1*2+0] = W[2]
            # sW row 3 -> c22 (k=1, i=1) -> W[1*2+1] = W[3]
            w_mapping = [0, 1, 2, 3]
            for i in range(4):
                W[w_mapping[i]][col] = sW[i][c]
        
        col_offset += 7
    
    # Process single last column (k=12)
    k = 12
    for c in range(4):
        col = col_offset + c
        
        # U: same (4 entries)
        for i in range(4):
            U[i][col] = bU[i][c]
        
        # V: rows j*p+k for j=0,1
        # bV row 0 -> e (j=0, k=12) -> V[0*13+12] = V[12]
        # bV row 1 -> f (j=1, k=12) -> V[1*13+12] = V[25]
        v_mapping = [12, 25]
        for i in range(2):
            V[v_mapping[i]][col] = bV[i][c]
        
        # W: rows k*n+i for k=0 (only k=0 since p=1 in this subproblem)
        # Wait, the output C has p=13 columns. For this block, we're computing
        # A * B[:, 12] which gives a 2x1 result.
        # C[0][12] -> index in W: k*n+i where k=12, i=0 -> W[12*2+0] = W[24]
        # C[1][12] -> index in W: k*n+i where k=12, i=1 -> W[12*2+1] = W[25]
        w_mapping = [24, 25]
        for i in range(2):
            W[w_mapping[i]][col] = bW[i][c]
    
    col_offset += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
