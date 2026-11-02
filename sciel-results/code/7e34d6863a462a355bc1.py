"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block-partitioned strategy:
    # Split p=8 into two p=4 blocks (columns 0-3 and 4-7 of B).
    # For each 2x3x4 sub-tensor, split m=3 into m=2 and m=1.
    #   - m=2, p=4: split p=4 into two p=2 blocks, each a 2x2x2 (Strassen, rank 7). Total rank 14.
    #   - m=1, p=4: schoolbook 2x1x4, rank 8.
    # Total rank per p=4 block: 14 + 8 = 22.
    # Total rank for p=8: 22 + 22 = 44.
    
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen rank-7 decomposition for 2x2x2 at offsets (i0,j0,k0)."""
        nonlocal col
        # m1: (a+d)(e+h) -> C[0,0] += 1, C[1,1] += 1
        U[(i0+0)*m + (j0+0)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # m2: (c+d)e -> C[1,0] += 1, C[1,1] += -1
        U[(i0+1)*m + (j0+0)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += 1
        W[(k0+0)*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1
        # m3: a(f-h) -> C[0,1] += 1, C[1,1] += 1
        U[(i0+0)*m + (j0+0)][col] += 1
        V[(j0+0)*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += -1
        W[(k0+1)*n + (i0+0)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # m4: d(g-e) -> C[0,0] += 1, C[1,0] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+0)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += -1
        W[(k0+0)*n + (i0+0)][col] += 1
        W[(k0+0)*n + (i0+1)][col] += 1
        col += 1
        # m5: (a+b)h -> C[0,0] += -1, C[0,1] += 1
        U[(i0+0)*m + (j0+0)][col] += 1
        U[(i0+0)*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += -1
        W[(k0+1)*n + (i0+0)][col] += 1
        col += 1
        # m6: (c-a)(e+f) -> C[1,1] += 1
        U[(i0+1)*m + (j0+0)][col] += 1
        U[(i0+0)*m + (j0+0)][col] += -1
        V[(j0+0)*p + (k0+0)][col] += 1
        V[(j0+0)*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # m7: (b-d)(g+h) -> C[0,0] += 1
        U[(i0+0)*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += -1
        V[(j0+1)*p + (k0+0)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += 1
        col += 1

    def add_schoolbook_2x1x4(i0, j0, k0):
        """Add schoolbook rank-8 decomposition for 2x1x4 at offsets (i0,j0,k0)."""
        nonlocal col
        for ii in range(2):
            for kk in range(4):
                U[(i0+ii)*m + (j0+0)][col] = 1
                V[(j0+0)*p + (k0+kk)][col] = 1
                W[(k0+kk)*n + (i0+ii)][col] = 1
                col += 1

    # Process each p=4 block
    for block in range(2):
        k_base = block * 4
        # m=2 part: two 2x2x2 Strassen blocks
        add_strassen_2x2x2(0, 0, k_base + 0)
        add_strassen_2x2x2(0, 0, k_base + 2)
        # m=1 part: schoolbook 2x1x4
        add_schoolbook_2x1x4(0, 2, k_base)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
