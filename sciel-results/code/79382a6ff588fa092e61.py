def add_optimized_3x3(i0, j0, k0):
    """Add optimized decomposition for 3x3 block using Strassen for 2x2 sub-block."""
    # Strassen for the 2x2 sub-block at local indices (0,1)
    strassen_2x2(i0, i0+1, j0, j0+1, k0, k0+1)
    
    # Remaining terms involving local index 2
    # A[0,2]*B[2,0] -> C[0,0]
    add_term({(i0, j0+2): 1}, {(j0+2, k0): 1}, {(k0, i0): 1})
    # A[0,2]*B[2,1] -> C[1,0]
    add_term({(i0, j0+2): 1}, {(j0+2, k0+1): 1}, {(k0+1, i0): 1})
    # A[0,2]*B[2,2] -> C[2,0]
    add_term({(i0, j0+2): 1}, {(j0+2, k0+2): 1}, {(k0+2, i0): 1})
    # A[1,2]*B[2,0] -> C[0,1]
    add_term({(i0+1, j0+2): 1}, {(j0+2, k0): 1}, {(k0, i0+1): 1})
    # A[1,2]*B[2,1] -> C[1,1]
    add_term({(i0+1, j0+2): 1}, {(j0+2, k0+1): 1}, {(k0+1, i0+1): 1})
    # A[1,2]*B[2,2] -> C[2,1]
    add_term({(i0+1, j0+2): 1}, {(j0+2, k0+2): 1}, {(k0+2, i0+1): 1})
    # A[2,0]*B[0,2] -> C[0,2]
    add_term({(i0+2, j0): 1}, {(j0, k0+2): 1}, {(k0+2, i0+2): 1})
    # A[2,0]*B[1,2] -> C[1,2]
    add_term({(i0+2, j0): 1}, {(j0+1, k0+2): 1}, {(k0+2, i0+2): 1})
    # A[2,0]*B[2,2] -> C[2,2]
    add_term({(i0+2, j0): 1}, {(j0+2, k0+2): 1}, {(k0+2, i0+2): 1})
    # A[2,1]*B[0,2] -> C[0,2]
    add_term({(i0+2, j0+1): 1}, {(j0, k0+2): 1}, {(k0+2, i0+2): 1})
    # A[2,1]*B[1,2] -> C[1,2]
    add_term({(i0+2, j0+1): 1}, {(j0+1, k0+2): 1}, {(k0+2, i0+2): 1})
    # A[2,1]*B[2,2] -> C[2,2]
    add_term({(i0+2, j0+1): 1}, {(j0+2, k0+2): 1}, {(k0+2, i0+2): 1})
    # A[2,2]*B[2,2] -> C[2,2]
    add_term({(i0+2, j0+2): 1}, {(j0+2, k0+2): 1}, {(k0+2, i0+2): 1})
