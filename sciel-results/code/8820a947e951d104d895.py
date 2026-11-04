"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 12 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 16 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 12 rows
    
    col = 0
    
    def strassen_2x2(offset_a, offset_b, offset_c):
        """Add Strassen decomposition for 2x2x2x2. offset_a/b/c are [i00,i01,i10,i11]."""
        nonlocal col
        a00, a01, a10, a11 = offset_a
        b00, b01, b10, b11 = offset_b
        c00, c01, c10, c11 = offset_c
        
        # M1 = (a00+a11)(b00+b11) -> contributes to c00, c11
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00 -> contributes to c10
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M3 = a00(b01-b11) -> contributes to c01
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M4 = a11(b10-b00) -> contributes to c10
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11 -> contributes to c00, c01
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1; W[c01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01) -> contributes to c11
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11) -> contributes to c11
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c11][col] += 1
        col += 1
    
    # A1 (rows 0,1) * B decomposition
    # A1_left: U[0,1,4,5] (i*m+j for i=0,1 j=0,1)
    # A1_right: U[2,3,6,7] (i*m+j for i=0,1 j=2,3)
    # B_top_left: V[0,1,4,5] (j*p+k for j=0,1 k=0,1)
    # B_top_right: V[2,3,6,7] (j*p+k for j=0,1 k=2,3)
    # B_bottom_left: V[8,9,12,13] (j*p+k for j=2,3 k=0,1)
    # B_bottom_right: V[10,11,14,15] (j*p+k for j=2,3 k=2,3)
    # C[0,0:2]: W[0,3] (k*n+i for k=0,1 i=0)
    # C[1,0:2]: W[1,4] (k*n+i for k=0,1 i=1)
    # C[0,2:4]: W[6,9] (k*n+i for k=2,3 i=0)
    # C[1,2:4]: W[7,10] (k*n+i for k=2,3 i=1)
    
    # A1_left * B_top_left -> C[0,0:2], C[1,0:2]
    strassen_2x2([0,1,4,5], [0,1,4,5], [0,3,1,4])
    # A1_left * B_top_right -> C[0,2:4], C[1,2:4]
    strassen_2x2([0,1,4,5], [2,3,6,7], [6,9,7,10])
    # A1_right * B_bottom_left -> C[0,0:2], C[1,0:2]
    strassen_2x2([2,3,6,7], [8,9,12,13], [0,3,1,4])
    # A1_right * B_bottom_right -> C[0,2:4], C[1,2:4]
    strassen_2x2([2,3,6,7], [10,11,14,15], [6,9,7,10])
    
    # A2 (row 2) * B: direct, 16 terms
    # A2: U[8,9,10,11] (i*m+j for i=2 j=0..3)
    # B: V[0..15]
    # C[2,k]: W[k*3+2] for k=0..3 -> W[2,5,8,11]
    for j in range(4):
        for k in range(4):
            U[8 + j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
