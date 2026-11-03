"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Split p=8 into four p=2 blocks.
    # For each p=2 block, split m=3 into m=2 and m=1.
    # m=2, p=2: 2x2x2, rank 7 (Strassen)
    # m=1, p=2: 2x1x2, rank 1 (single rank-1 term)
    # Total per p=2 block: 7 + 1 = 8
    # Total rank: 4 * 8 = 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # Strassen 7 multiplications for 2x2x2
        # U: (i, j) -> idx = i*m + j
        # V: (j, k) -> idx = j*p + k
        # W: (k, i) -> idx = k*n + i
        
        # m1: (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m2: (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # m3: a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m4: d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # m5: (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1
        
        # m6: (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m7: (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    def add_rank1_2x1x2(i_off, j_off, k_off):
        nonlocal col
        # 2x1x2 has rank 1: T_{i,k,0} = 1
        # U[i, j_off] = 1 for i in {0,1}
        # V[j_off, k] = 1 for k in {0,1}
        # W[k, i] = 1 for all i,k
        for i in range(2):
            U[(i_off+i)*m + j_off][col] += 1
        for k in range(2):
            V[j_off*p + (k_off+k)][col] += 1
        for k in range(2):
            for i in range(2):
                W[(k_off+k)*n + (i_off+i)][col] += 1
        col += 1

    # Process each p=2 block
    for k_off in range(0, 8, 2):
        # m=2 part (j=0,1): Strassen 2x2x2
        add_strassen_2x2x2(0, 0, k_off)
        # m=1 part (j=2): rank-1 2x1x2
        add_rank1_2x1x2(0, 2, k_off)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
