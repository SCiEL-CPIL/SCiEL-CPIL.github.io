import numpy as np

def construct():
    n, m, p = 2, 3, 4
    # Rank 15 decomposition
    # Part 1: A[0:2, 0:2] * B[0:2, :] (2x2x4) using two 2x2x2 Strassen blocks
    # Part 2: A[0:2, 2:3] * B[2:3, :] (2x1x4) using 1 rank-1 term
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)] # 6x15
    V = [[0] * rank for _ in range(m * p)] # 12x15
    W = [[0] * rank for _ in range(p * n)] # 8x15
    
    # Helper to add a Strassen block for 2x2x2
    # offset_u: row offset in U (i*m + j)
    # offset_v: row offset in V (j*p + k)
    # offset_w: row offset in W (k*n + i)
    # col_offset: column offset in U, V, W
    def add_strassen(offset_u, offset_v, offset_w, col_offset):
        # Strassen coefficients for 2x2x2
        # A indices: 0,1,2,3 (row 0: 0,1; row 1: 2,3)
        # B indices: 0,1,2,3 (col 0: 0,1; col 1: 2,3) -- Wait, B is 2x2.
        # B indices in V: j*p + k.
        # For a 2x2 block of B at rows j0, j0+1 and cols k0, k0+1.
        # V indices: j0*p + k0, j0*p + k0+1, (j0+1)*p + k0, (j0+1)*p + k0+1.
        
        # M1 = (a+d)(e+h)
        # a: A[0,0], d: A[1,1]
        # e: B[0,0], h: B[1,1]
        U[offset_u + 0][col_offset + 0] += 1
        U[offset_u + 3][col_offset + 0] += 1
        V[offset_v + 0][col_offset + 0] += 1
        V[offset_v + 3][col_offset + 0] += 1
        # C[0,0] += M1, C[1,1] += M1
        W[offset_w + 0][col_offset + 0] += 1 # C[0,0] index 0*2+0=0
        W[offset_w + 3][col_offset + 0] += 1 # C[1,1] index 1*2+1=3
        
        # M2 = (c+d)e
        # c: A[1,0], d: A[1,1]
        # e: B[0,0]
        U[offset_u + 2][col_offset + 1] += 1
        U[offset_u + 3][col_offset + 1] += 1
        V[offset_v + 0][col_offset + 1] += 1
        # C[1,0] += M2
        W[offset_w + 2][col_offset + 1] += 1 # C[1,0] index 0*2+1=2
        
        # M3 = a(f-h)
        # a: A[0,0]
        # f: B[0,1], h: B[1,1]
        U[offset_u + 0][col_offset + 2] += 1
        V[offset_v + 1][col_offset + 2] += 1
        V[offset_v + 3][col_offset + 2] += -1
        # C[0,1] += M3
        W[offset_w + 1][col_offset + 2] += 1 # C[0,1] index 1*2+0=1
        
        # M4 = d(g-e)
        # d: A[1,1]
        # g: B[1,0], e: B[0,0]
        U[offset_u + 3][col_offset + 3] += 1
        V[offset_v + 2][col_offset + 3] += 1
        V[offset_v + 0][col_offset + 3] += -1
        # C[0,0] -= M4, C[0,1] += M4
        W[offset_w + 0][col_offset + 3] += -1
        W[offset_w + 1][col_offset + 3] += 1
        
        # M5 = (a+b)h
        # a: A[0,0], b: A[0,1]
        # h: B[1,1]
        U[offset_u + 0][col_offset + 4] += 1
        U[offset_u + 1][col_offset + 4] += 1
        V[offset_v + 3][col_offset + 4] += 1
        # C[0,0] += M5, C[1,0] += M5
        W[offset_w + 0][col_offset + 4] += 1
        W[offset_w + 2][col_offset + 4] += 1
        
        # M6 = (c-a)(e+f)
        # c: A[1,0], a: A[0,0]
        # e: B[0,0], f: B[0,1]
        U[offset_u + 2][col_offset + 5] += 1
        U[offset_u + 0][col_offset + 5] += -1
        V[offset_v + 0][col_offset + 5] += 1
        V[offset_v + 1][col_offset + 5] += 1
        # C[0,0] += M6, C[1,0] += M6
        W[offset_w + 0][col_offset + 5] += 1
        W[offset_w + 2][col_offset + 5] += 1
        
        # M7 = (b-d)(g+h)
        # b: A[0,1], d: A[1,1]
        # g: B[1,0], h: B[1,1]
        U[offset_u + 1][col_offset + 6] += 1
        U[offset_u + 3][col_offset + 6] += -1
        V[offset_v + 2][col_offset + 6] += 1
        V[offset_v + 3][col_offset + 6] += 1
        # C[1,1] -= M7
        W[offset_w + 3][col_offset + 6] += -1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # A rows 0,1. Cols 0,1.
    # U offset: 0 (i=0, j=0 -> 0; i=0, j=1 -> 1; i=1, j=0 -> 2; i=1, j=1 -> 3)
    # V offset: 0 (j=0, k=0 -> 0; j=0, k=1 -> 1; j=1, k=0 -> 2; j=1, k=1 -> 3)
    # W offset: 0 (k=0, i=0 -> 0; k=0, i=1 -> 1; k=1, i=0 -> 2; k=1, i=1 -> 3)
    add_strassen(0, 0, 0, 0)

    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    # A same.
    # B rows 0,1. Cols 2,3.
    # V indices: j=0, k=2 -> 2; j=0, k=3 -> 3; j=1, k=2 -> 6; j=1, k=3 -> 7.
    # Wait, V index is j*p + k. p=4.
    # j=0, k=2 -> 2. j=0, k=3 -> 3.
    # j=1, k=2 -> 1*4+2=6. j=1, k=3 -> 1*4+3=7.
    # So V offset is not contiguous 0..3.
    # My add_strassen assumes contiguous 0..3 for B.
    # I need to map the local B indices to global V indices.
    
    # Let's modify add_strassen to take a mapping for V indices.
    # Or just write it out.
    
    # Let's restart the construction with explicit indices to be safe.
    
    U = [[0] * 15 for _ in range(6)]
    V = [[0] * 15 for _ in range(12)]
    W = [[0] * 15 for _ in range(8)]
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # U indices: 0,1,2,3
    # V indices: 0,1,2,3
    # W indices: 0,1,2,3
    
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    
    # M3
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[1][2] += 1
    
    # M4
    U[3][3] += 1
    V[2][3] += 1; V[0][3] += -1
    W[0][3] += -1; W[1][3] += 1
    
    # M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += 1; W[2][4] += 1
    
    # M6
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[0][5] += 1; W[2][5] += 1
    
    # M7
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[3][6] += -1
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # U indices: 0,1,2,3
    # V indices: 2,3,6,7 (j=0,k=2; j=0,k=3; j=1,k=2; j=1,k=3)
    # W indices: 4,5,6,7 (k=2,i=0; k=2,i=1; k=3,i=0; k=3,i=1)
    # k=2 -> 2*2+0=4, 2*2+1=5.
    # k=3 -> 3*2+0=6, 3*2+1=7.
    
    # M1
    U[0][7] += 1; U[3][7] += 1
    V[2][7] += 1; V[7][7] += 1
    W[4][7] += 1; W[7][7] += 1
    
    # M2
    U[2][8] += 1; U[3][8] += 1
    V[2][8] += 1
    W[6][8] += 1 # C[1,2] index 2*2+1=5? No.
    # C[1,2] is row 1, col 2. Index 2*2+1 = 5.
    # Wait, W index is k*n + i.
    # C[1,2] -> k=2, i=1 -> 2*2+1 = 5.
    # My previous W index for C[1,0] was 2 (k=0, i=1 -> 0*2+1=1? No.
    # W[k*n + i].
    # C[0,0] -> k=0, i=0 -> 0.
    # C[0,1] -> k=1, i=0 -> 1*2+0=2.
    # C[1,0] -> k=0, i=1 -> 0*2+1=1.
    # C[1,1] -> k=1, i=1 -> 1*2+1=3.
    
    # Let's re-verify W indices for Block 1.
    # C[0,0] (k=0, i=0) -> 0.
    # C[0,1] (k=1, i=0) -> 2.
    # C[1,0] (k=0, i=1) -> 1.
    # C[1,1] (k=1, i=1) -> 3.
    
    # Block 1 W assignments:
    # M1: C[0,0] (0), C[1,1] (3). Correct.
    # M2: C[1,0] (1). My code had W[2][1]. Error.
    # M3: C[0,1] (2). My code had W[1][2]. Error.
    # M4: C[0,0] (0), C[0,1] (2). My code had W[0], W[1]. Error.
    # M5: C[0,0] (0), C[1,0] (1). My code had W[0], W[2]. Error.
    # M6: C[0,0] (0), C[1,0] (1). My code had W[0], W[2]. Error.
    # M7: C[1,1] (3). Correct.
    
    # I need to fix the W indices.
    
    # Let's define a helper to get W index.
    def w_idx(k, i): return k * n + i
    
    # Re-do Block 1
    # M2: C[1,0] -> w_idx(0,1) = 1.
    # M3: C[0,1] -> w_idx(1,0) = 2.
    # M4: C[0,0] (0), C[0,1] (2).
    # M5: C[0,0] (0), C[1,0] (1).
    # M6: C[0,0] (0), C[1,0] (1).
    
    # Block 2 W indices:
    # C[0,2] (k=2, i=0) -> 4.
    # C[0,3] (k=3, i=0) -> 6.
    # C[1,2] (k=2, i=1) -> 5.
    # C[1,3] (k=3, i=1) -> 7.
    
    # Block 2 assignments:
    # M1: C[0,2] (4), C[1,3] (7).
    # M2: C[1,2] (5).
    # M3: C[0,3] (6).
    # M4: C[0,2] (4), C[0,3] (6).
    # M5: C[0,2] (4), C[1,2] (5).
    # M6: C[0,2] (4), C[1,2] (5).
    # M7: C[1,3] (7).
    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # U indices: 4,5 (i=0,j=2; i=1,j=2).
    # V indices: 8,9,10,11 (j=2, k=0..3).
    # W indices: 0..7.
    # Rank 1 term.
    # U[4][14] += 1; U[5][14] += 1
    # V[8][14] += 1; V[9][14] += 1; V[10][14] += 1; V[11][14] += 1
    # W[0..7][14] += 1
    
    # Let's implement this cleanly.

    U = [[0] * 15 for _ in range(6)]
    V = [[0] * 15 for _ in range(12)]
    W = [[0] * 15 for _ in range(8)]
    
    # Block 1 (cols 0-1 of B)
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    # M2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1
    # M3
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[2][2] += 1
    # M4
    U[3][3] += 1
    V[2][3] += 1; V[0][3] += -1
    W[0][3] += -1; W[2][3] += 1
    # M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += 1; W[1][4] += 1
    # M6
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[0][5] += 1; W[1][5] += 1
    # M7
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[3][6] += -1
    
    # Block 2 (cols 2-3 of B)
    # M1
    U[0][7] += 1; U[3][7] += 1
    V[2][7] += 1; V[7][7] += 1
    W[4][7] += 1; W[7][7] += 1
    # M2
    U[2][8] += 1; U[3][8] += 1
    V[2][8] += 1
    W[5][8] += 1
    # M3
    U[0][9] += 1
    V[3][9] += 1; V[7][9] += -1
    W[6][9] += 1
    # M4
    U[3][10] += 1
    V[6][10] += 1; V[2][10] += -1
    W[4][10] += -1; W[6][10] += 1
    # M5
    U[0][11] += 1; U[1][11] += 1
    V[7][11] += 1
    W[4][11] += 1; W[5][11] += 1
    # M6
    U[2][12] += 1; U[0][12] += -1
    V[2][12] += 1; V[3][12] += 1
    W[4][12] += 1; W[5][12] += 1
    # M7
    U[1][13] += 1; U[3][13] += -1
    V[6][13] += 1; V[7][13] += 1
    W[7][13] += -1
    
    # Block 3 (col 2 of A, row 2 of B)
    U[4][14] += 1; U[5][14] += 1
    V[8][14] += 1; V[9][14] += 1; V[10][14] += 1; V[11][14] += 1
    for k in range(4):
        for i in range(2):
            W[k*2 + i][14] += 1
            
    return {"U": U, "V": V, "W": W}
