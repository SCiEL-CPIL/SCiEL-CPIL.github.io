"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank for 4x4 is 49
    rank = 49
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen patterns for 2x2 matrices
    # Each pattern is a tuple of (U_dict, V_dict, W_dict)
    # Keys are (i,j), (j,k), (k,i) respectively for 2x2 indices (0,1)
    
    # M0: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    P0 = (
        {(0,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,1): 1}
    )
    # M1: (A10+A11)B00 -> C01 += 1, C11 -= 1
    P1 = (
        {(1,0): 1, (1,1): 1}, 
        {(0,0): 1}, 
        {(0,1): 1, (1,1): -1}
    )
    # M2: A00(B01-B11) -> C10 += 1, C11 += 1
    P2 = (
        {(0,0): 1}, 
        {(0,1): 1, (1,1): -1}, 
        {(1,0): 1, (1,1): 1}
    )
    # M3: A11(B10-B00) -> C00 += 1, C01 += 1
    P3 = (
        {(1,1): 1}, 
        {(1,0): 1, (0,0): -1}, 
        {(0,0): 1, (0,1): 1}
    )
    # M4: (A00+A01)B11 -> C00 -= 1, C10 += 1
    P4 = (
        {(0,0): 1, (0,1): 1}, 
        {(1,1): 1}, 
        {(0,0): -1, (1,0): 1}
    )
    # M5: (A10-A00)(B00+B01) -> C00 += 1, C10 += 1
    P5 = (
        {(1,0): 1, (0,0): -1}, 
        {(0,0): 1, (0,1): 1}, 
        {(0,0): 1, (1,0): 1}
    )
    # M6: (A01-A11)(B10+B11) -> C11 += 1
    P6 = (
        {(0,1): 1, (1,1): -1}, 
        {(1,0): 1, (1,1): 1}, 
        {(1,1): 1}
    )
    
    patterns = [P0, P1, P2, P3, P4, P5, P6]
    
    # Iterate over the 49 combinations (7 top-level * 7 bottom-level)
    for t1 in range(7):
        U1, V1, W1 = patterns[t1]
        for t2 in range(7):
            U2, V2, W2 = patterns[t2]
            r = t1 * 7 + t2
            
            # Expand indices
            # i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
            for i1, j1 in U1:
                u_val1 = U1[(i1, j1)]
                for i0, j0 in U2:
                    u_val2 = U2[(i0, j0)]
                    u_total = u_val1 * u_val2
                    i = 2 * i1 + i0
                    j = 2 * j1 + j0
                    U[i * m + j][r] = u_total
            
            for j1, k1 in V1:
                v_val1 = V1[(j1, k1)]
                for j0, k0 in V2:
                    v_val2 = V2[(j0, k0)]
                    v_total = v_val1 * v_val2
                    j = 2 * j1 + j0
                    k = 2 * k1 + k0
                    V[j * p + k][r] = v_total
            
            for k1, i1 in W1:
                w_val1 = W1[(k1, i1)]
                for k0, i0 in W2:
                    w_val2 = W2[(k0, i0)]
                    w_total = w_val1 * w_val2
                    k = 2 * k1 + k0
                    i = 2 * i1 + i0
                    W[k * n + i][r] = w_total
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
