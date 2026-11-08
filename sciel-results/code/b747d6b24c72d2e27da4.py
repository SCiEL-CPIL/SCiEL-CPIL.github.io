"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a term
    # u_idx: i*m + j
    # v_idx: j*p + k
    # w_idx: k*n + i
    # coeff: value
    
    def set_term(r, i, j, k, val):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][r] += val
        V[v_idx][r] += val
        W[w_idx][r] += val

    # 1. Strassen Part (7 terms) for j in {0,1}, k in {0,1}
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    r = 0
    set_term(r, 0, 0, 0, 1)
    set_term(r, 1, 1, 0, 1) # Wait, u is (i,j), v is (j,k).
    # M1: u: (0,0), (1,1). v: (0,0), (1,1).
    # My set_term adds to specific (i,j,k).
    # Strassen term is a product of sums.
    # U[r] has 1s at (0,0) and (1,1).
    # V[r] has 1s at (0,0) and (1,1).
    # W[r] has 1s at (0,0) and (1,1) for C00, C11.
    # But C00 corresponds to i=0, k=0. C11 corresponds to i=1, k=1.
    # So W[r] should have 1 at (0,0) and 1 at (1,1).
    # The tensor term is sum over i,j,k.
    # If U has 1 at (0,0) and (1,1), and V has 1 at (0,0) and (1,1),
    # the product is A00B00 + A00B11 + A11B00 + A11B11.
    # This matches (A00+A11)(B00+B11).
    # Then W must map this to C00 and C11.
    # C00 = i=0, k=0. C11 = i=1, k=1.
    # So W[r] should have 1 at (0,0) and 1 at (1,1).
    
    # Let's re-verify the Strassen decomposition mapping.
    # We define 7 columns r=0..6.
    
    # r=0: M1
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*3+0][0] = 1; V[1*3+1][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*3+0][1] = 1
    W[1*2+0][1] = 1; W[1*2+1][1] = -1 # C10 += M2, C11 -= M2
    
    # r=2: M3 = A00(B01-B11)
    U[0*3+0][2] = 1
    V[0*3+1][2] = 1; V[1*3+1][2] = -1
    W[0*2+1][2] = 1; W[1*2+1][2] = 1 # C01 += M3, C11 += M3
    
    # r=3: M4 = A11(B10-B00)
    U[1*3+1][3] = 1
    V[1*3+0][3] = 1; V[0*3+0][3] = -1
    W[0*2+0][3] = 1; W[1*2+0][3] = 1 # C00 += M4, C10 += M4
    
    # r=4: M5 = (A00+A01)B11
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*3+1][4] = 1
    W[0*2+0][4] = -1; W[0*2+1][4] = 1 # C00 -= M5, C01 += M5
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[0*3+0][5] = 1; V[0*3+1][5] = 1
    W[1*2+1][5] = 1 # C11 += M6
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[0*3+1][6] = 1; U[1*3+1][6] = -1
    V[1*3+0][6] = 1; V[1*3+1][6] = 1
    W[0*2+0][6] = 1 # C00 += M7

    # 2. Edge Part 1: j in {0,1}, k=2 (Rank 4)
    # r=7: A00 B02 -> C02
    U[0*3+0][7] = 1; V[0*3+2][7] = 1; W[2*2+0][7] = 1
    # r=8: A01 B12 -> C02
    U[0*3+1][8] = 1; V[1*3+2][8] = 1; W[2*2+0][8] = 1
    # r=9: A10 B02 -> C12
    U[1*3+0][9] = 1; V[0*3+2][9] = 1; W[2*2+1][9] = 1
    # r=10: A11 B12 -> C12
    U[1*3+1][10] = 1; V[1*3+2][10] = 1; W[2*2+1][10] = 1

    # 3. Edge Part 2: j=2, k in {0,1} (Rank 4)
    # r=11: A02 B20 -> C00
    U[0*3+2][11] = 1; V[2*3+0][11] = 1; W[0*2+0][11] = 1
    # r=12: A02 B21 -> C01
    U[0*3+2][12] = 1; V[2*3+1][12] = 1; W[1*2+0][12] = 1
    # r=13: A12 B20 -> C10
    U[1*3+2][13] = 1; V[2*3+0][13] = 1; W[0*2+1][13] = 1
    # r=14: A12 B21 -> C11
    U[1*3+2][14] = 1; V[2*3+1][14] = 1; W[1*2+1][14] = 1

    # 4. Edge Part 3: j=2, k=2 (Rank 2)
    # r=15: A02 B22 -> C02
    U[0*3+2][15] = 1; V[2*3+2][15] = 1; W[2*2+0][15] = 1
    # r=16: A12 B22 -> C12
    U[1*3+2][16] = 1; V[2*3+2][16] = 1; W[2*2+1][16] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
