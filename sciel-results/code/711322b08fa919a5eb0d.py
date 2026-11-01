"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * (7 + 4) = 44
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to the decomposition
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Iterate over the 4 blocks of B (3x2 each)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k2]
        # A indices: 0,1,3,4 (row*m+col)
        # B indices: 0*p+k0, 0*p+k1, 1*p+k0, 1*p+k1
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        add_term(0, 1, 0*p+k0, 1, k0*n+0, 1)
        U[1*m+1][col-1] = 1
        V[1*p+k1][col-1] = 1
        W[k1*n+1][col-1] = 1

        # M2 = (A10 + A11) * B0k0
        add_term(1*m+0, 1, 0*p+k0, 1, k0*n+1, 1)
        U[1*m+1][col-1] = 1
        W[k1*n+1][col-1] = -1

        # M3 = A00 * (B0k1 - B1k1)
        add_term(0, 1, 0*p+k1, 1, k1*n+0, 1)
        V[1*p+k1][col-1] = -1
        W[k1*n+1][col-1] = 1

        # M4 = A11 * (B1k0 - B0k0)
        add_term(1*m+1, 1, 1*p+k0, 1, k0*n+0, 1)
        V[0*p+k0][col-1] = -1
        W[k0*n+1][col-1] = 1

        # M5 = (A00 + A01) * B1k1
        add_term(0, 1, 1*p+k1, 1, k0*n+0, -1)
        U[0*m+1][col-1] = 1
        W[k1*n+0][col-1] = 1

        # M6 = (A10 - A00) * (B0k0 + B0k1)
        add_term(1*m+0, 1, 0*p+k0, 1, k1*n+1, 1)
        U[0*m+0][col-1] = -1
        V[0*p+k1][col-1] = 1

        # M7 = (A01 - A11) * (B1k0 + B1k1)
        add_term(0*m+1, 1, 1*p+k0, 1, k0*n+0, 1)
        U[1*m+1][col-1] = -1
        V[1*p+k1][col-1] = 1

        # Rank-4 part for A[0:2, 2] * B[2, k0:k2]
        # A indices: 2, 5
        # B indices: 2*p+k0, 2*p+k1
        
        # P1 = A02 * B2k0
        add_term(2, 1, 2*p+k0, 1, k0*n+0, 1)

        # P2 = A02 * B2k1
        add_term(2, 1, 2*p+k1, 1, k1*n+0, 1)

        # P3 = A12 * B2k0
        add_term(1*m+2, 1, 2*p+k0, 1, k0*n+1, 1)

        # P4 = A12 * B2k1
        add_term(1*m+2, 1, 2*p+k1, 1, k1*n+1, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
