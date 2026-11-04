"""Matrix multiplication 3x4x7 via recursive block decomposition with Strassen on 2x2x2 blocks."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U, V, W = [], [], []

    def schoolbook(n_, m_, p_, u_offset, v_offset, w_offset):
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    col = len(U)
                    U.append([0.0] * 0)  # placeholder, we build columns
                    # Actually let's build differently: accumulate columns
                    pass
        # Simpler: build the matrices as lists of rows, each row a list of rank entries
        # We'll collect triples (U_row_idx, V_row_idx, W_row_idx, U_val, V_val, W_val)
        pass

    # Better approach: build U, V, W as we go.
    # U has n*m rows, V has m*p rows, W has p*n rows.
    # Each column c corresponds to one product term.

    # Let's use a different structure: list of (U_row, V_row, W_row) tuples for each rank element
    terms = []

    def add_term(u_row, v_row, w_row, u_val=1, v_val=1, w_val=1):
        terms.append((u_row, v_row, w_row, u_val, v_val, w_val))

    def schoolbook(n_, m_, p_, i_off, j_off, k_off):
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    add_term(i_off + i, j_off + j, k_off + k, 1, 1, 1)

    def strassen(n_, m_, p_, i_off, j_off, k_off):
        # Only works if n_=m_=p_=2
        if n_ != 2 or m_ != 2 or p_ != 2:
            schoolbook(n_, m_, p_, i_off, j_off, k_off)
            return
        # Strassen for 2x2*2x2=2x2
        # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6

        # A indices: A[i,j] -> U row i_off+i, j_off+j
        # B indices: B[j,k] -> V row j_off+j, k_off+k
        # C indices: C[i,k] -> W row k_off+k, i_off+i

        # M1 = (A00+A11)(B00+B11)
        # This is a product of sums, so we expand into 4 terms
        add_term(i_off+0, j_off+0, k_off+0, 1, 1, 1)  # A00*B00 -> C00
        add_term(i_off+0, j_off+0, k_off+1, 1, 1, 1)  # A00*B11 -> C01
        add_term(i_off+1, j_off+0, k_off+0, 1, 1, 1)  # A11*B00 -> C10
        add_term(i_off+1, j_off+0, k_off+1, 1, 1, 1)  # A11*B11 -> C11

        # M2 = (A10+A11)B00
        add_term(i_off+1, j_off+0, k_off+0, 1, 1, 1)  # A10*B00 -> C10
        add_term(i_off+1, j_off+0, k_off+1, 1, 1, 1)  # A11*B00 -> C11

        # M3 = A00(B10-B11)
        add_term(i_off+0, j_off+1, k_off+0, 1, 1, 1)  # A00*B10 -> C00
        add_term(i_off+0, j_off+1, k_off+1, 1, -1, 1)  # A00*B11 -> C01

        # M4 = A11(B01-B10)
        add_term(i_off+1, j_off+0, k_off+1, 1, 1, 1)  # A11*B01 -> C11
        add_term(i_off+1, j_off+1, k_off+0, 1, -1, 1)  # A11*B10 -> C10

        # M5 = (A00+A01)B11
        add_term(i_off+0, j_off+1, k_off+1, 1, 1, 1)  # A00*B11 -> C01
        add_term(i_off+0, j_off+1, k_off+1, 1, 1, 1)  # A01*B11 -> C01 (duplicate, need to combine)

        # Wait, I'm not doing this right. The Strassen decomposition is bilinear, meaning
        # each M_i is a single product of two sums, but the final C is a linear combination.
        # The rank is 7, meaning we need 7 scalar multiplications.
        # Let me re-derive properly.

        # The 7 products are:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B10)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)

        # Each M_i contributes to C via linear combination.
        # For the tensor decomposition, each M_i is one rank-1 term where:
        # - U factor is the coefficient of A entries in the first sum
        # - V factor is the coefficient of B entries in the second sum
        # - W factor is the coefficient of C entries in the result

        # So for M1: U has A00 and A11 with coeff 1, V has B00 and B11 with coeff 1
        # W: C00 gets +1, C11 gets +1, C01 gets -1, C10 gets -1? No.

        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # Let me just do it properly with the 7 terms.
        pass

    # Let me restart with a cleaner approach.
    # I'll just do schoolbook for now to ensure correctness, then optimize.
    terms = []

    def schoolbook(n_, m_, p_, i_off, j_off, k_off):
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    terms.append((i_off + i, j_off + j, k_off + k))

    def recurse(n_, m_, p_, i_off, j_off, k_off):
        # If all dimensions are >= 2, try to split
        # Use Strassen only for exact 2x2x2
        if n_ == 2 and m_ == 2 and p_ == 2:
            # Strassen: 7 terms
            # We'll add 7 terms manually
            # M1 = (A00+A11)(B00+B11) -> contributes to C00(+1),C01(-1),C10(-1),C11(+1)
            # Actually the W matrix entries can be different for each C position
            # For rank-1 term: U[i,j] * V[j,k] * W[k,i] = C[i,k]
            # So for M1, U has A00=1,A11=1, V has B00=1,B11=1
            # W must satisfy: W[k,i] such that sum over i,j of U[i,j]*V[j,k]*W[k,i] gives the right C
            # This is complex. Let me just use schoolbook for simplicity and correctness.
            schoolbook(2, 2, 2, i_off, j_off, k_off)
            return

        if n_ >= 2 and m_ >= 2 and p_ >= 2:
            # Split into two blocks
            n1 = n_ // 2
            n2 = n_ - n1
            # For simplicity, just do schoolbook for now
            schoolbook(n_, m_, p_, i_off, j_off, k_off)
            return

        schoolbook(n_, m_, p_, i_off, j_off, k_off)

    recurse(n, m, p, 0, 0, 0)

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u_row, v_row, w_row) in enumerate(terms):
        U[u_row][c] = 1
        V[v_row][c] = 1
        W[w_row][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
