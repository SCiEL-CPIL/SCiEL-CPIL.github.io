"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: i=0, schoolbook for 1x4x8
    # U index: 0*m + j = j
    # V index: j*p + k = 8j + k
    # W index: k*n + 0 = 3k
    for j in range(m):
        for k in range(p):
            r = j * p + k
            U[j][r] = 1
            V[j * p + k][r] = 1
            W[k * n][r] = 1
            
    # Part 2: i=1,2, Strassen for 2x4x8 split into 4 blocks of 2x2x2
    # U indices for i=1,2:
    # j=0 -> 4, j=1 -> 5, j=2 -> 6, j=3 -> 7 (i=1)
    # j=0 -> 8, j=1 -> 9, j=2 -> 10, j=3 -> 11 (i=2)
    
    # V indices for j=0..3, k=0..7:
    # j=0 -> 0..7
    # j=1 -> 8..15
    # j=2 -> 16..23
    # j=3 -> 24..31
    
    # W indices for i=1,2, k=0..7:
    # k=0 -> 3*0+1=1, 3*0+2=2
    # k=1 -> 3*1+1=4, 3*1+2=5
    # ...
    
    for c in range(4):
        r_base = 32 + c * 14
        
        # Left part: A[1:3, 0:2] * B[0:2, 2c:2c+2]
        # U rows: 4, 5, 8, 9
        u00, u01, u10, u11 = 4, 5, 8, 9
        # V rows: 2c, 2c+1, 8+2c, 8+2c+1
        v00, v01, v10, v11 = 2*c, 2*c+1, 8+2*c, 8+2*c+1
        # W rows: 6c+1, 6c+4, 6c+2, 6c+5
        w00, w01, w10, w11 = 6*c+1, 6*c+4, 6*c+2, 6*c+5
        
        # M1
        U[u00][r_base+0] += 1
        U[u11][r_base+0] += 1
        V[v00][r_base+0] += 1
        V[v11][r_base+0] += 1
        W[w00][r_base+0] += 1
        W[w11][r_base+0] += 1
        
        # M2
        U[u10][r_base+1] += 1
        U[u11][r_base+1] += 1
        V[v00][r_base+1] += 1
        W[w10][r_base+1] += 1
        W[w11][r_base+1] += -1
        
        # M3
        U[u00][r_base+2] += 1
        V[v11][r_base+2] += 1
        V[v01][r_base+2] += -1
        W[w01][r_base+2] += 1
        W[w11][r_base+2] += 1
        
        # M4
        U[u11][r_base+3] += 1
        V[v10][r_base+3] += 1
        V[v00][r_base+3] += -1
        W[w00][r_base+3] += 1
        W[w10][r_base+3] += 1
        
        # M5
        U[u00][r_base+4] += 1
        U[u01][r_base+4] += 1
        V[v11][r_base+4] += 1
        W[w00][r_base+4] += -1
        W[w01][r_base+4] += 1
        
        # M6
        U[u10][r_base+5] += 1
        U[u00][r_base+5] += -1
        V[v00][r_base+5] += 1
        V[v01][r_base+5] += 1
        W[w11][r_base+5] += 1
        
        # M7
        U[u01][r_base+6] += 1
        U[u11][r_base+6] += -1
        V[v10][r_base+6] += 1
        V[v11][r_base+6] += 1
        W[w00][r_base+6] += 1
        
        # Right part: A[1:3, 2:4] * B[2:4, 2c:2c+2]
        # U rows: 6, 7, 10, 11
        u00, u01, u10, u11 = 6, 7, 10, 11
        # V rows: 16+2c, 16+2c+1, 24+2c, 24+2c+1
        v00, v01, v10, v11 = 16+2*c, 16+2*c+1, 24+2*c, 24+2*c+1
        # W rows: same as left
        # w00, w01, w10, w11 already defined
        
        # M1'
        U[u00][r_base+7] += 1
        U[u11][r_base+7] += 1
        V[v00][r_base+7] += 1
        V[v11][r_base+7] += 1
        W[w00][r_base+7] += 1
        W[w11][r_base+7] += 1
        
        # M2'
        U[u10][r_base+8] += 1
        U[u11][r_base+8] += 1
        V[v00][r_base+8] += 1
        W[w10][r_base+8] += 1
        W[w11][r_base+8] += -1
        
        # M3'
        U[u00][r_base+9] += 1
        V[v11][r_base+9] += 1
        V[v01][r_base+9] += -1
        W[w01][r_base+9] += 1
        W[w11][r_base+9] += 1
        
        # M4'
        U[u11][r_base+10] += 1
        V[v10][r_base+10] += 1
        V[v00][r_base+10] += -1
        W[w00][r_base+10] += 1
        W[w10][r_base+10] += 1
        
        # M5'
        U[u00][r_base+11] += 1
        U[u01][r_base+11] += 1
        V[v11][r_base+11] += 1
        W[w00][r_base+11] += -1
        W[w01][r_base+11] += 1
        
        # M6'
        U[u10][r_base+12] += 1
        U[u00][r_base+12] += -1
        V[v00][r_base+12] += 1
        V[v01][r_base+12] += 1
        W[w11][r_base+12] += 1
        
        # M7'
        U[u01][r_base+13] += 1
        U[u11][r_base+13] += -1
        V[v10][r_base+13] += 1
        V[v11][r_base+13] += 1
        W[w00][r_base+13] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
