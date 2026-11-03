"""Matrix multiplication 2x3x8 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    # Strategy: Block-partitioned decomposition
    # Split p=8 into four p=2 blocks.
    # For each p=2 block, split m=3 into m=2 and m=1.
    # m=2, p=2 part is 2x2x2, rank 7 (Strassen).
    # m=1, p=2 part is 2x1x2, rank 4 (Schoolbook).
    # Total rank per p=2 block: 7 + 4 = 11.
    # Total rank for p=8: 4 * 11 = 44.
    
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # Local U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global U index: (i_off + li) * m + (j_off + lj)
        # Local V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global V index: (j_off + lj) * p + (k_off + lk)
        # Local W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
        # Global W index: (k_off + lk) * n + (i_off + li)
        
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

    def add_schoolbook_2x1x2(i_off, j_off, k_off):
        nonlocal col
        # 2x1x2 schoolbook: 4 terms
        # i in 0..1, k in 0..1
        for i in range(2):
            for k in range(2):
                U[(i_off+i)*m + (j_off+0)][col] = 1
                V[(j_off+0)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Iterate over 4 blocks of p=2
    for block in range(4):
        k_off = block * 2
        
        # m=2 part (j=0,1)
        add_strassen_2x2x2(0, 0, k_off)
        
        # m=1 part (j=2)
        add_schoolbook_2x1x2(0, 2, k_off)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
