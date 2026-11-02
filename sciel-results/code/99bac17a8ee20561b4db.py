"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # A: i (0..1), j (0..2) -> idx = i*3 + j
    # B: j (0..2), k (0..2) -> idx = j*3 + k
    # C: i (0..1), k (0..2) -> idx = k*2 + i (W[k*n+i])
    
    col = 0
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # B indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M3 = A00(B10-B11)
    U[0][col] = 1
    V[3][col] = 1; V[4][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M4 = A11(B01-B00)
    U[4][col] = 1
    V[1][col] = 1; V[0][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # Part 2: Tail A[0:2, 2] * B[2, 0:3] (Outer Product)
    # A indices: (0,2)->2, (1,2)->5
    # B indices: (2,0)->6, (2,1)->7, (2,2)->8
    # C indices: (0,0)->0, (0,1)->1, (0,2)->2, (1,0)->3, (1,1)->4, (1,2)->5
    
    # i=0, k=0
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1
    col += 1
    # i=0, k=1
    U[2][col] = 1; V[7][col] = 1; W[1][col] = 1
    col += 1
    # i=0, k=2
    U[2][col] = 1; V[8][col] = 1; W[2][col] = 1
    col += 1
    # i=1, k=0
    U[5][col] = 1; V[6][col] = 1; W[3][col] = 1
    col += 1
    # i=1, k=1
    U[5][col] = 1; V[7][col] = 1; W[4][col] = 1
    col += 1
    # i=1, k=2
    U[5][col] = 1; V[8][col] = 1; W[5][col] = 1
    col += 1
    
    # Part 3: A[0:2, 0:2] * B[0:2, 2] (2x2x1)
    # A indices: 0,1,3,4
    # B indices: (0,2)->2, (1,2)->5
    # C indices: (0,2)->2, (1,2)->5
    
    # Rank of 2x2x1 is 2?
    # If I use 2 terms, I need to find them.
    # If I use 4 terms, it's schoolbook.
    # Let's use 2 terms if possible.
    # Actually, I will use 2 terms from a known decomposition.
    # M8 = A00 * B02 + A11 * B12 ? No.
    
    # I will use the 2 terms that cover the cross terms?
    # Let's just use 2 terms:
    # M8 = (A00 + A11) * (B02 + B12)
    # M9 = (A01 + A10) * (B02 - B12) ?
    
    # If I can't verify, I'll use 4 terms (rank 19 total? No, 18-1=17? No 7+6+4=17).
    # 17 is better than 18.
    
    # Let's use 4 terms for safety (Rank 17).
    # i=0, j=0
    U[0][col] = 1; V[2][col] = 1; W[2][col] = 1
    col += 1
    # i=0, j=1
    U[1][col] = 1; V[5][col] = 1; W[2][col] = 1
    col += 1
    # i=1, j=0
    U[3][col] = 1; V[2][col] = 1; W[5][col] = 1
    col += 1
    # i=1, j=1
    U[4][col] = 1; V[5][col] = 1; W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
