"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Part 1: Columns 0, 1 (Strassen)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: i*m+j
    # 0:00, 1:01, 2:10, 3:11
    # B indices: j*p+k
    # 0:00, 1:01, 2:02, 3:03, 4:04
    # 5:10, 6:11, 7:12, 8:13, 9:14
    # C indices: k*n+i
    # 0:00, 1:10
    # 2:01, 3:11
    # 4:02, 5:12
    # 6:03, 7:13
    # 8:04, 9:14

    col = 0
    
    # M1
    U[0][col] += 1; U[3][col] += 1
    V[0][col] += 1; V[6][col] += 1
    W[0][col] += 1; W[1][col] += 1; W[2][col] += -1; W[3][col] += -1
    col += 1
    
    # M2
    U[2][col] += 1; U[3][col] += 1
    V[0][col] += 1
    W[1][col] += 1; W[3][col] += -1
    col += 1
    
    # M3
    U[0][col] += 1
    V[5][col] += 1; V[6][col] += -1
    W[2][col] += 1; W[3][col] += 1
    col += 1
    
    # M4
    U[3][col] += 1
    V[1][col] += 1; V[0][col] += -1
    W[0][col] += 1; W[1][col] += 1
    col += 1
    
    # M5
    U[0][col] += 1; U[1][col] += 1
    V[6][col] += 1
    W[0][col] += -1; W[2][col] += 1
    col += 1
    
    # M6
    U[2][col] += 1; U[0][col] += -1
    V[0][col] += 1; V[1][col] += 1
    W[1][col] += -1; W[3][col] += 1
    col += 1
    
    # M7
    U[1][col] += 1; U[3][col] += -1
    V[5][col] += 1; V[6][col] += 1
    W[0][col] += 1; W[2][col] += 1; W[1][col] += -1; W[3][col] += -1
    col += 1

    # Part 2: Columns 2, 3 (Strassen)
    # Same A coefficients, B coeffs shifted to k=2,3, W coeffs shifted to k=2,3
    
    # M1
    U[0][col] += 1; U[3][col] += 1
    V[2][col] += 1; V[8][col] += 1
    W[4][col] += 1; W[5][col] += 1; W[6][col] += -1; W[7][col] += -1
    col += 1
    
    # M2
    U[2][col] += 1; U[3][col] += 1
    V[2][col] += 1
    W[5][col] += 1; W[7][col] += -1
    col += 1
    
    # M3
    U[0][col] += 1
    V[7][col] += 1; V[8][col] += -1
    W[6][col] += 1; W[7][col] += 1
    col += 1
    
    # M4
    U[3][col] += 1
    V[3][col] += 1; V[2][col] += -1
    W[4][col] += 1; W[5][col] += 1
    col += 1
    
    # M5
    U[0][col] += 1; U[1][col] += 1
    V[8][col] += 1
    W[4][col] += -1; W[6][col] += 1
    col += 1
    
    # M6
    U[2][col] += 1; U[0][col] += -1
    V[2][col] += 1; V[3][col] += 1
    W[5][col] += -1; W[7][col] += 1
    col += 1
    
    # M7
    U[1][col] += 1; U[3][col] += -1
    V[7][col] += 1; V[8][col] += 1
    W[4][col] += 1; W[6][col] += 1; W[5][col] += -1; W[7][col] += -1
    col += 1

    # Part 3: Column 4 (Schoolbook)
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    
    # A00 B04
    U[0][col] += 1
    V[4][col] += 1
    W[8][col] += 1
    col += 1
    
    # A01 B14
    U[1][col] += 1
    V[9][col] += 1
    W[8][col] += 1
    col += 1
    
    # A10 B04
    U[2][col] += 1
    V[4][col] += 1
    W[9][col] += 1
    col += 1
    
    # A11 B14
    U[3][col] += 1
    V[9][col] += 1
    W[9][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
