"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    # A is n x m (3x4), B is m x p (4x5), C is n x p (3x5)
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (15, rank)
    # W[k*n+i, r] where k is row of C (0..p-1), i is col of C (0..n-1)
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Indexing:
    # U[i*m + j] = A[i][j]
    # V[j*p + k] = B[j][k]
    # W[k*n + i] = C[i][k] (row i, col k)
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if coeff != 0:
            U[u_idx][col] += coeff
            V[v_idx][col] += coeff
            W[w_idx][col] += coeff
        col += 1
    
    def add_strassen(a_i0, a_j0, b_j0, b_k0):
        """Add Strassen 2x2*2x2 decomposition.
        A sub-block: rows a_i0..a_i0+1, cols a_j0..a_j0+1
        B sub-block: rows b_j0..b_j0+1, cols b_k0..b_k0+1
        C sub-block: rows a_i0..a_i0+1, cols b_k0..b_k0+1
        """
        # A entries:
        # a = A[a_i0][a_j0], b = A[a_i0][a_j0+1]
        # c = A[a_i0+1][a_j0], d = A[a_i0+1][a_j0+1]
        # B entries:
        # e = B[b_j0][b_k0], f = B[b_j0][b_k0+1]
        # g = B[b_j0+1][b_k0], h = B[b_j0+1][b_k0+1]
        # C entries:
        # C[a_i0][b_k0] = P1+P4-P5+P7
        # C[a_i0][b_k0+1] = P3+P5
        # C[a_i0+1][b_k0] = P2+P4
        # C[a_i0+1][b_k0+1] = P1-P2+P3+P6
        
        # P1 = (a+d)(e+h)
        add_term(a_i0*m + a_j0, b_j0*p + b_k0, (b_k0)*n + a_i0, 1)
        U[a_i0*m + a_j0+1][col-1] += 0  # no, we need to add both a and d
        # Actually, add_term adds coeff to one U, one V, one W.
        # For P1 = (a+d)(e+h), we need U to have 1 at both a and d positions.
        # Let me redo this more carefully.
        col -= 1  # undo the add_term
        # P1: U has 1 at a and d, V has 1 at e and h
        U[a_i0*m + a_j0][col] += 1
        U[(a_i0+1)*m + (a_j0+1)][col] += 1
        V[b_j0*p + b_k0][col] += 1
        V[(b_j0+1)*p + (b_k0+1)][col] += 1
        W[b_k0*n + a_i0][col] += 1
        W[(b_k0+1)*n + (a_i0+1)][col] += 1
        col += 1
        
        # P2 = (c+d)*e
        U[(a_i0+1)*m + a_j0][col] += 1
        U[(a_i0+1)*m + (a_j0+1)][col] += 1
        V[b_j0*p + b_k0][col] += 1
        W[b_k0*n + (a_i0+1)][col] += 1
        col += 1
        
        # P3 = a*(f-h)
        U[a_i0*m + a_j0][col] += 1
        V[b_j0*p + (b_k0+1)][col] += 1
        V[(b_j0+1)*p + (b_k0+1)][col] -= 1
        W[(b_k0+1)*n + a_i0][col] += 1
        col += 1
        
        # P4 = d*(g-e)
        U[(a_i0+1)*m + (a_j0+1)][col] += 1
        V[(b_j0+1)*p + b_k0][col] += 1
        V[b_j0*p + b_k0][col] -= 1
        W[b_k0*n + (a_i0+1)][col] += 1
        W[(b_k0+1)*n + (a_i0+1)][col] += 1
        col += 1
        
        # P5 = (a+b)*h
        U[a_i0*m + a_j0][col] += 1
        U[a_i0*m + (a_j0+1)][col] += 1
        V[(b_j0+1)*p + (b_k0+1)][col] += 1
        W[(b_k0+1)*n + a_i0][col] += 1
        W[(b_k0+1)*n + (a_i0+1)][col] -= 1
        col += 1
        
        # P6 = (c-a)*(e+f)
        U[(a_i0+1)*m + a_j0][col] += 1
        U[a_i0*m + a_j0][col] -= 1
        V[b_j0*p + b_k0][col] += 1
        V[b_j0*p + (b_k0+1)][col] += 1
        W[(b_k0+1)*n + (a_i0+1)][col] += 1
        col += 1
        
        # P7 = (b-d)*(g+h)
        U[a_i0*m + (a_j0+1)][col] += 1
        U[(a_i0+1)*m + (a_j0+1)][col] -= 1
        V[(b_j0+1)*p + b_k0][col] += 1
        V[(b_j0+1)*p + (b_k0+1)][col] += 1
        W[b_k0*n + a_i0][col] += 1
        W[(b_k0+1)*n + a_i0][col] += 1
        col += 1
    
    # Decomposition:
    # Split A (3x4) into A1 (3x2, cols 0-1) and A2 (3x2, cols 2-3)
    # Split B (4x5) into B1 (2x5, rows 0-1) and B2 (2x5, rows 2-3)
    # C = A1*B1 + A2*B2
    
    # For each half (A1*B1 and A2*B2):
    #   A_half is 3x2, B_half is 2x5
    #   Split A_half into top row (1x2) and bottom 2x2
    #   Split B_half into left 2x2 and right 2x3
    #   top_row * B_half: 1x2 * 2x5, schoolbook rank 10
    #   bottom_2x2 * left_2x2: Strassen rank 7
    #   bottom_2x2 * right_2x3: schoolbook rank 8
    #   Total per half: 25
    
    # Half 1: A1 (rows 0-2, cols 0-1) * B1 (rows 0-1, cols 0-4)
    # Top row of A1: A[0][0], A[0][1]
    # B1: rows 0-1, cols 0-4
    # 1x2 * 2x5 schoolbook:
    for j in range(2):  # A[0][j], j=0,1
        for k in range(5):  # B[j][k]
            add_term(0*m + j, j*p + k, k*n + 0)
    # That's 10 terms
    
    # Bottom 2x2 of A1: A[1][0],A[1][1],A[2][0],A[2][1]
    # Left 2x2 of B1: B[0][0],B[0][1],B[1][0],B[1][1]
    # Strassen:
    add_strassen(1, 0, 0, 0)
    # That's 7 terms
    
    # Bottom 2x2 of A1: A[1][0],A[1][1],A[2][0],A[2][1]
    # Right 2x3 of B1: B[0][2],B[0][3],B[0][4],B[1][2],B[1][3],B[1][4]
    # Schoolbook 2x2 * 2x3:
    for i in range(2):  # A rows 1,2
        for j in range(2):  # A cols 0,1
            for k in range(3):  # B cols 2,3,4
                add_term((1+i)*m + j, j*p + (2+k), (2+k)*n + (1+i))
    # That's 8 terms
    
    # Half 2: A2 (rows 0-2, cols 2-3) * B2 (rows 2-3, cols 0-4)
    # Top row of A2: A[0][2], A[0][3]
    # B2: rows 2-3, cols 0-4
    # 1x2 * 2x5 schoolbook:
    for j in range(2, 4):  # A[0][j], j=2,3
        for k in range(5):  # B[j][k]
            add_term(0*m + j, j*p + k, k*n + 0)
    # That's 10 terms
    
    # Bottom 2x2 of A2: A[1][2],A[1][3],A[2][2],A[2][3]
    # Left 2x2 of B2: B[2][0],B[2][1],B[3][0],B[3][1]
    # Strassen:
    add_strassen(1, 2, 2, 0)
    # That's 7 terms
    
    # Bottom 2x2 of A2: A[1][2],A[1][3],A[2][2],A[2][3]
    # Right 2x3 of B2: B[2][2],B[2][3],B[2][4],B[3][2],B[3][3],B[3][4]
    # Schoolbook 2x2 * 2x3:
    for i in range(2):  # A rows 1,2
        for j in range(2, 4):  # A cols 2,3
            for k in range(3):  # B cols 2,3,4
                add_term((1+i)*m + j, j*p + (2+k), (2+k)*n + (1+i))
    # That's 8 terms
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
