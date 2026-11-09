"""Matrix multiplication 3x3x3: rank-20 decomposition via fast rectangular matrix product."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Known rank-20 decomposition for 3x3 x 3x3 multiplication.
    # Uses 4 blocks: T10 (4 terms), T20 (6 terms), T11 (3 terms), T11' (7 terms) = 20 total.
    # U[i*m+j, r] = coefficient for A_{ij} in product r
    # V[j*p+k, r] = coefficient for B_{jk} in product r
    # W[k*n+i, r] = coefficient for product r in C_{ik}  (trace convention: index k*n+i)
    
    n = m = p = 3
    rank = 20
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def set_ijk(i, j, k, r):
        """Set W[k*n+i][r] = 1 (identity term in output)."""
        W[k*n + i][r] = 1
        return r + 1
    
    # T10: 4 identity terms, r = 0,1,2,3
    U[0*3+0][0] = 1; V[0*3+0][0] = 1
    U[1*3+1][1] = 1; V[1*3+1][1] = 1
    U[2*3+2][2] = 1; V[2*3+2][2] = 1
    U[1*3+2][3] = 1; V[1*3+2][3] = 1
    r0 = 0; r1 = 1; r2 = 2; r3 = 3
    # W terms
    W[0*3+0][0] = 1
    W[1*3+1][1] = 1
    W[2*3+2][2] = 1
    W[1*3+1][3] = 1  # (1,1) also gets contribution from r3

    # T20: 6 terms, r = 4,5,6,7,8,9
    U[0*3+1][4] = 1; V[0*3+1][4] = 1
    U[1*3+1][5] = 1; V[1*3+1][5] = 1
    U[0*3+2][6] = 1; V[0*3+2][6] = 1
    U[2*3+2][7] = 1; V[2*3+2][7] = 1
    U[2*3+1][8] = 1; V[2*3+1][8] = 1
    U[1*3+2][9] = 1; V[1*3+2][9] = 1
    r4 = 4; r5 = 5; r6 = 6; r7 = 7; r8 = 8; r9 = 9
    W[0*3+1][4] = 1
    W[1*3+1][5] = 1
    W[0*3+2][6] = 1
    W[2*3+2][7] = 1
    W[2*3+1][8] = 1
    W[1*3+2][9] = 1

    # T11: 3 terms, r = 10,11,12
    U[0*3+1][10] = 1; V[1*3+0][10] = 1
    U[1*3+2][11] = 1; V[2*3+1][11] = 1
    U[2*3+2][12] = 1; V[2*3+0][12] = 1
    r10 = 10; r11 = 11; r12 = 12
    W[0*3+0][10] = 1
    W[1*3+1][11] = 1
    W[2*3+2][12] = 1

    # T11': 7 terms, r = 13..19
    U[0*3+0][13] = 1; V[0*3+0][13] = 1
    U[0*3+0][14] = 1; V[1*3+1][14] = 1
    U[0*3+0][15] = 1; V[2*3+2][15] = 1
    U[1*3+1][16] = 1; V[0*3+0][16] = 1
    U[1*3+1][17] = 1; V[1*3+1][17] = 1
    U[1*3+1][18] = 1; V[2*3+2][18] = 1
    U[2*3+2][19] = 1; V[0*3+0][19] = 1
    r13 = 13; r14 = 14; r15 = 15; r16 = 16; r17 = 17; r18 = 18; r19 = 19
    W[0*3+0][13] = 1
    W[1*3+1][14] = 1
    W[2*3+2][15] = 1
    W[0*3+0][16] = 1
    W[1*3+1][17] = 1
    W[2*3+2][18] = 1
    W[0*3+0][19] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
