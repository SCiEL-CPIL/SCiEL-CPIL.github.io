import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # 27 for 3x3x3 (schoolbook)
    # 7 for 2x2x2 (Strassen) part of 3x3x2
    # 4 for 1x2x2 (schoolbook) part of 3x3x2
    # 6 for 3x1x2 (schoolbook) part of 3x3x2
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: 3x3x3 Schoolbook (cols 0,1,2 of B)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 5 + k
                w_idx = k * 3 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    # Block 2: 3x3x2 (cols 3,4 of B)
    # Sub-block 2a: 2x2x2 Strassen (A rows 0,1; B rows 0,1; C rows 0,1; C cols 3,4)
    # A indices: i in {0,1}, j in {0,1} -> u_idx = i*3+j
    # B indices: j in {0,1}, k in {3,4} -> v_idx = j*5+k
    # C indices: i in {0,1}, k in {3,4} -> w_idx = k*3+i
    
    # M1 = (A00+A11)(B03+B14)
    U[0][col] = 1; U[4][col] = 1
    V[3][col] = 1; V[9][col] = 1
    W[9][col] = 1; W[13][col] = 1
    col += 1
    
    # M2 = (A10+A11)B03
    U[3][col] = 1; U[4][col] = 1
    V[3][col] = 1
    W[10][col] = 1; W[13][col] = -1
    col += 1
    
    # M3 = A00(B13-B03)
    U[0][col] = 1
    V[8][col] = 1; V[3][col] = -1
    W[12][col] = 1; W[13][col] = 1
    col += 1
    
    # M4 = A11(B14-B04)
    U[4][col] = 1
    V[9][col] = 1; V[4][col] = -1
    W[9][col] = -1; W[12][col] = 1
    col += 1
    
    # M5 = (A00+A01)B14
    U[0][col] = 1; U[1][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B03+B04)
    U[3][col] = 1; U[0][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[13][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B13+B14)
    U[1][col] = 1; U[4][col] = -1
    V[8][col] = 1; V[9][col] = 1
    W[10][col] = 1; W[9][col] = -1
    col += 1
    
    # Sub-block 2b: 1x2x2 Schoolbook (A row 2; B rows 0,1; C row 2; C cols 3,4)
    # A indices: i=2, j in {0,1} -> u_idx = 2*3+j = 6+j
    # B indices: j in {0,1}, k in {3,4} -> v_idx = j*5+k
    # C indices: i=2, k in {3,4} -> w_idx = k*3+2
    
    # A20 B03 -> C23
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    # A20 B04 -> C24
    U[6][col] = 1
    V[4][col] = 1
    W[14][col] = 1
    col += 1
    
    # A21 B13 -> C23
    U[7][col] = 1
    V[8][col] = 1
    W[11][col] = 1
    col += 1
    
    # A21 B14 -> C24
    U[7][col] = 1
    V[9][col] = 1
    W[14][col] = 1
    col += 1
    
    # Sub-block 2c: 3x1x2 Schoolbook (A col 2; B row 2; C all rows; C cols 3,4)
    # A indices: i in {0,1,2}, j=2 -> u_idx = i*3+2
    # B indices: j=2, k in {3,4} -> v_idx = 2*5+k = 10+k
    # C indices: i in {0,1,2}, k in {3,4} -> w_idx = k*3+i
    
    # A02 B23 -> C03
    U[2][col] = 1
    V[13][col] = 1
    W[9][col] = 1
    col += 1
    
    # A02 B24 -> C04
    U[2][col] = 1
    V[14][col] = 1
    W[12][col] = 1
    col += 1
    
    # A12 B23 -> C13
    U[5][col] = 1
    V[13][col] = 1
    W[10][col] = 1
    col += 1
    
    # A12 B24 -> C14
    U[5][col] = 1
    V[14][col] = 1
    W[13][col] = 1
    col += 1
    
    # A22 B23 -> C23
    U[8][col] = 1
    V[13][col] = 1
    W[11][col] = 1
    col += 1
    
    # A22 B24 -> C24
    U[8][col] = 1
    V[14][col] = 1
    W[14][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
