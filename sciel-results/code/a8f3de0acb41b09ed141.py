"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Local indices for Strassen part (k in {0,1}, j in {0,1}, i in {0,1})
    # U[i, k, r], V[k, j, r], W[j, i, r]
    
    # r=0 (M1)
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*10+0][0] = 1; V[1*10+1][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    
    # r=1 (M2)
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*10+0][1] = 1
    W[0*2+1][1] = 1; W[1*2+1][1] = -1
    
    # r=2 (M3)
    U[0*3+0][2] = 1
    V[0*10+1][2] = 1; V[1*10+1][2] = -1
    W[1*2+0][2] = 1; W[1*2+1][2] = 1
    
    # r=3 (M4)
    U[1*3+1][3] = 1
    V[1*10+0][3] = 1; V[0*10+0][3] = -1
    W[0*2+0][3] = 1; W[0*2+1][3] = 1
    
    # r=4 (M5)
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*10+1][4] = 1
    W[0*2+0][4] = -1; W[1*2+0][4] = 1
    
    # r=5 (M6)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[0*10+0][5] = 1; V[0*10+1][5] = 1
    W[1*2+1][5] = 1
    
    # r=6 (M7)
    U[0*3+1][6] = 1; U[1*3+1][6] = -1
    V[1*10+0][6] = 1; V[1*10+1][6] = 1
    W[0*2+0][6] = 1
    
    # Trivial part for k=2 (r=7..10)
    # r=7: A[0,2] B[2,0] -> C[0,0]
    U[0*3+2][7] = 1
    V[2*10+0][7] = 1
    W[0*2+0][7] = 1
    
    # r=8: A[0,2] B[2,1] -> C[0,1]
    U[0*3+2][8] = 1
    V[2*10+1][8] = 1
    W[1*2+0][8] = 1
    
    # r=9: A[1,2] B[2,0] -> C[1,0]
    U[1*3+2][9] = 1
    V[2*10+0][9] = 1
    W[0*2+1][9] = 1
    
    # r=10: A[1,2] B[2,1] -> C[1,1]
    U[1*3+2][10] = 1
    V[2*10+1][10] = 1
    W[1*2+1][10] = 1
    
    # Copy this block structure to all 5 blocks of p=2
    # The above definitions used local j=0,1.
    # We need to shift V and W indices for each block.
    # U indices do not depend on j (block), so they are the same for all blocks?
    # No, U is shared across blocks?
    # The rank is additive. Each block uses distinct columns in U, V, W.
    # So for block b, we use columns b*11 + r_local.
    
    # My initialization above only filled columns 0-10.
    # I need to fill columns for b=1..4 as well.
    
    for b in range(1, 5):
        offset = b * 11
        # Copy U entries (same for all blocks)
        for i in range(n * m):
            for r_local in range(11):
                U[i][offset + r_local] = U[i][r_local]
                
        # Shift V and W indices
        # V index: k * p + j. j is local 0,1. Global j = 2*b + j_local.
        # V[k, j_local, r_local] -> V[k, 2*b + j_local, offset + r_local]
        for k in range(m):
            for j_local in range(2):
                v_idx_local = k * p + j_local
                v_idx_global = k * p + (2 * b + j_local)
                for r_local in range(11):
                    V[v_idx_global][offset + r_local] = V[v_idx_local][r_local]
                    
        # W index: j * n + i. j is local 0,1. Global j = 2*b + j_local.
        # W[j_local, i, r_local] -> W[2*b + j_local, i, offset + r_local]
        for i in range(n):
            for j_local in range(2):
                w_idx_local = j_local * n + i
                w_idx_global = (2 * b + j_local) * n + i
                for r_local in range(11):
                    W[w_idx_global][offset + r_local] = W[w_idx_local][r_local]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
