"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank-28 exact decomposition for 2x3 * 3x5 using Strassen on 2x2 blocks.
    # Structure: split 5 columns of B into 2 and 3.
    #   Part 1: M_{2,3,2} rank 11 (Strassen 2x2x2 + schoolbook 2x1x2)
    #   Part 2: M_{2,3,3} rank 17 (M_{2,2,3} rank 11 + schoolbook 2x1x3)
    #     M_{2,2,3} rank 11 (Strassen 2x2x2 + schoolbook 2x2x1)
    #
    # U shape: (6, 28)  rows = i*m+j for i in 0..1, j in 0..2
    # V shape: (15, 28) rows = j*p+k for j in 0..2, k in 0..4
    # W shape: (10, 28) rows = k*n+i for k in 0..4, i in 0..1
    #
    # We build the decomposition block by block.

    n, m, p = 2, 3, 5
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def add_strassen_222(u_offset, v_offset, w_offset, base_col):
        """Add a Strassen 2x2x2 decomposition.
        u_offset: (i_base, j_base) offset into A (n x m sub-block)
        v_offset: (j_base, k_base) offset into B (m x p sub-block)
        w_offset: (k_base, i_base) offset into C (p x n sub-block)
        base_col: starting column in U, V, W
        Uses 7 Strassen products:
          P1 = (A11+A22)(B11+B22)
          P2 = (A21+A22)B11
          P3 = A11(B21-B22)
          P4 = A22(B12-B11)
          P5 = (A11+A12)B22
          P6 = (A21-A11)(B11+B12)
          P7 = (A12-A22)(B21+B22)
        C11 = P1 + P4 - P5 + P7
        C12 = P3 + P5
        C21 = P2 + P4
        C22 = P1 - P2 + P3 + P6
        """
        i0, j0 = u_offset
        j1, k0 = v_offset
        k1, i1 = w_offset

        # Strassen product assignments to columns base_col .. base_col+6
        # For each product r in 0..6:
        #   U row: for each (i, j) in A's 2x2 block, set U[i0+i, j0+j] += coeff
        #   V row: for each (j, k) in B's 2x2 block, set V[j1+j, k0+k] += coeff
        #   W row: for each (k, i) in C's 2x2 block, set W[k1+k, i1+i] += coeff

        # P1 = (A11+A22)(B11+B22): A factors: (1,0)->1, (1,1)->1; B factors: (0,0)->1, (1,1)->1
        # Actually Strassen: each product is a bilinear form.
        # U gets coefficients for A, V for B, W for C.
        # The rank-1 term is: sum_{i,j} U[i,j]*t_r * sum_{j',k} V[j',k]*s_r = sum_{k',i'} W[k',i'] * (A*B term)
        # Standard form: C = sum_r (U_r * A) (B * V_r) where U_r, V_r are 2x2 matrices of coefficients
        # and W_r is a 2x2 matrix.
        #
        # Let me use the standard Strassen formulation:
        # For product r, we have matrices Lr (2x2, left mult on A), Mr (2x2, right mult on B), Nr (2x2, adds to C)
        # C += Nr * (Lr * A) * (B * Mr) ... no that's not right either.
        #
        # The tensor decomposition: T[i,j,k,l,m,n] = sum_r U[i,j,r] * V[k,l,r] * W[m,n,r]
        # where the multiplication is T = A * B with A[i,j], B[k,l], C[m,n] and j=k, l=n... 
        # Wait, let me re-read the convention.
        # U shape (n*m, rank): row index = i*m + j (A is n x m)
        # V shape (m*p, rank): row index = j*p + k (B is m x p)
        # W shape (p*n, rank): row index = k*n + i (C is p x n)
        # Convention: W[k*n+i, r] -- this is the official trace convention.
        #
        # The decomposition: C[i, k] = sum_j A[i, j] * B[j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not quite right dimensionally. Let me think again.
        #
        # Actually the standard tensor decomposition for matrix multiplication:
        # C = A * B, where A is n x m, B is m x p, C is n x p.
        # T[i, j, k] = A[i,j] * B[j,k] = C[i,k] (with j summed)
        # But the output W is indexed by (k, i) with shape (p, n) = (p*n,).
        # So W[k*n + i, r] is the coefficient for C[i, k].
        #
        # The decomposition: C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        # No wait, that's not how it works either. Let me be more careful.
        #
        # The matrix multiplication tensor T has indices (i, j, k) for A[i,j] * B[j,k].
        # A rank-r decomposition: T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[i,k,r]
        # In our indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # So: A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        #
        # For Strassen on 2x2x2: A is 2x2 (i=0,1; j=0,1), B is 2x2 (j=0,1; k=0,1), C is 2x2 (i=0,1; k=0,1)
        # The 7 Strassen products give:
        # P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # P2 = (A[1,0]+A[1,1])*B[0,0]
        # P3 = A[0,0]*(B[1,0]-B[1,1])
        # P4 = A[1,1]*(B[0,1]-B[0,0])
        # P5 = (A[0,0]+A[0,1])*B[1,1]
        # P6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # P7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        #
        # C[0,0] = P1 + P4 - P5 + P7
        # C[0,1] = P3 + P5
        # C[1,0] = P2 + P4
        # C[1,1] = P1 - P2 + P3 + P6
        #
        # For each product r, we need U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = A[i,j]*B[j,k]
        #
        # This means for each r, the "A-part" is a linear combination of A entries,
        # the "B-part" is a linear combination of B entries, and the "C-part" assigns
        # the product to C entries.
        #
        # For P1: A-part = A[0,0] + A[1,1], B-part = B[0,0] + B[1,1], C-part: contributes to C[0,0] and C[1,1]
        #   U: U[0,0]+=1, U[1,1]+=1 (i*m+j: (0,0)->0, (1,1)->1*2+1=3)
        #   V: V[0,0]+=1, V[1,1]+=1 (j*p+k: (0,0)->0, (1,1)->1*2+1=3)
        #   W: W[0,0]+=1, W[1,1]+=1 (k*n+i: (0,0)->0, (1,1)->1*2+1=3)
        #
        # Wait, but the W contribution should be: P1 appears in C[0,0] with coeff +1 and C[1,1] with coeff +1.
        # So W[0*n+0, r] += 1 and W[1*n+1, r] += 1.
        #
        # Let me code this up properly.

        # Product definitions: (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: list of (i, j, coeff)
        # B_coeffs: list of (j, k, coeff)
        # C_coeffs: list of (i, k, coeff)
        products = [
            # P1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # P2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
            # P3
            ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1)]),
            # P4
            ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # P5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # P6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # P7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,-1)]),
        ]

        for r, (a_coefs, b_coefs, c_coefs) in enumerate(products):
            col = base_col + r
            for (ii, jj, coeff) in a_coefs:
                U[i0 + ii][j0 + jj] += coeff
            for (jj, kk, coeff) in b_coefs:
                V[j1 + jj][k0 + kk] += coeff
            for (ii, kk, coeff) in c_coefs:
                W[k1 + kk][i1 + ii] += coeff

    def add_schoolbook_212(u_offset, v_offset, w_offset, base_col):
        """Add a schoolbook 2x1x2 decomposition (A is 2x1, B is 1x2, C is 2x2).
        A[i,0] * B[0,k] -> C[i,k], 4 products.
        u_offset: (i_base, j_base) where j_base is the single column
        v_offset: (j_base, k_base) where j_base is the single row of B
        w_offset: (k_base, i_base)
        """
        i0, j0 = u_offset
        j1, k0 = v_offset
        k1, i1 = w_offset

        for r in range(4):
            col = base_col + r
            i = r // 2
            k = r % 2
            U[i0 + i][j0] += 1
            V[j1][k0 + k] += 1
            W[k1 + k][i1 + i] += 1

    def add_schoolbook_221(u_offset, v_offset, w_offset, base_col):
        """Add a schoolbook 2x2x1 decomposition (A is 2x2, B is 2x1, C is 2x1).
        C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0], 4 products.
        """
        i0, j0 = u_offset
        j1, k0 = v_offset
        k1, i1 = w_offset

        for r in range(4):
            col = base_col + r
            i = r // 2
            j = r % 2
            U[i0 + i][j0 + j] += 1
            V[j1 + j][k0] += 1
            W[k1][i1 + i] += 1

    def add_schoolbook_213(u_offset, v_offset, w_offset, base_col):
        """Add a schoolbook 2x1x3 decomposition (A is 2x1, B is 1x3, C is 2x3).
        6 products.
        """
        i0, j0 = u_offset
        j1, k0 = v_offset
        k1, i1 = w_offset

        for r in range(6):
            col = base_col + r
            i = r // 3
            k = r % 3
            U[i0 + i][j0] += 1
            V[j1][k0 + k] += 1
            W[k1 + k][i1 + i] += 1

    # Part 1: M_{2,3,2} (A cols 0-2, B rows 0-2, B cols 0-1)
    # Sub-part 1a: M_{2,2,2} (A cols 0-1, B rows 0-1, B cols 0-1)
    add_strassen_222((0, 0), (0, 0), (0, 0), 0)
    # Sub-part 1b: M_{2,1,2} (A col 2, B row 2, B cols 0-1)
    add_schoolbook_212((0, 2), (2, 0), (0, 0), 7)

    # Part 2: M_{2,3,3} (A cols 0-2, B rows 0-2, B cols 2-4)
    # Sub-part 2a: M_{2,2,3} (A cols 0-1, B rows 0-1, B cols 2-4)
    #   Sub-sub 2a1: M_{2,2,2} (A cols 0-1, B rows 0-1, B cols 2-3)
    add_strassen_222((0, 0), (0, 2), (2, 0), 11)
    #   Sub-sub 2a2: M_{2,2,1} (A cols 0-1, B rows 0-1, B col 4)
    add_schoolbook_221((0, 0), (0, 4), (4, 0), 18)
    # Sub-part 2b: M_{2,1,3} (A col 2, B row 2, B cols 2-4)
    add_schoolbook_213((0, 2), (2, 2), (2, 0), 22)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
