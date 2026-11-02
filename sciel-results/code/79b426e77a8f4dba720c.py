"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    # Rank-50 exact decomposition via block splitting:
    #   m=3 -> 1+2;  A_1 (2x1) * B_1 (1x9) : 18 terms
    #   A_2 (2x2) * B_2 (2x9): split p=9 into 4 blocks of 2 and one of 1
    #       4 * Strassen(2x2x2=7) + 1 * schoolbook(2x2x1=4) = 32 terms
    #   Total rank = 18 + 32 = 50
    n, m, p = DIMENSIONS  # 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # 6 rows
    V = [[0] * rank for _ in range(m * p)]   # 27 rows
    W = [[0] * rank for _ in range(p * n)]   # 18 rows

    col = 0

    # --- Block A_1 * B_1 : i in {0,1}, j=0, k in {0..8} : 2*9 = 18 terms ---
    for i in range(n):
        for k in range(9):
            U[i * m + 0][col] = 1          # U[i, 0]
            V[0 * p + k][col] = 1          # V[0, k]
            W[k * n + i][col] = 1          # W[k, i]
            col += 1
    # col is now 18

    # --- Block A_2 * B_2 : i in {0,1}, j in {1,2}, k in {0..8} ---
    # Split p=9 into chunks: [0:2], [2:4], [4:6], [6:8], [8:9]
    chunks = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 9)]
    for (ks, ke) in chunks:
        w = ke - ks  # width of this k-block (2 or 1)
        for i in range(n):
            for j in range(2):  # j=1,2 -> local j' = j
                for k_local in range(w):
                    k = ks + k_local
                    # U index: global row i*m + (1+j)
                    U[i * m + (1 + j)][col] = 1
                    # V index: global row (1+j)*p + k
                    V[(1 + j) * p + k][col] = 1
                    # W index: global row k*n + i
                    W[k * n + i][col] = 1
                    col += 1

    # Note: the above simple block decomposition for A_2*B_2 uses schoolbook
    # (2*2*9 = 36) instead of Strassen. Let me redo with Strassen for 2x2x2 blocks.

    # Reset and redo properly
    col = 0
    # --- Block A_1 * B_1 : 18 terms (unchanged) ---
    for i in range(n):
        for k in range(9):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # --- Block A_2 * B_2 with Strassen on 2x2x2 sub-blocks ---
    # A_2 is 2x2 (rows i=0,1; cols j=1,2), B_2 is 2x9 (rows j=1,2; cols k=0..8)
    # Split k into: [0,1], [2,3], [4,5], [6,7], [8]
    # For each 2x2x2 block use Strassen (7 mults); for 2x2x1 use schoolbook (4 mults)

    def strassen_block(k_start, j_start):
        """Add Strassen terms for A_2[i, j_start+j'] * B_2[j_start+j', k_start+k']
           where i in {0,1}, j' in {0,1}, k' in {0,1}.
           A_2 sub-block: A[0,0]=U[i=0,j=1], A[0,1]=U[i=0,j=2], A[1,0]=U[i=1,j=1], A[1,1]=U[i=1,j=2]
           B_2 sub-block: B[0,0]=V[j=1,k=ks], B[0,1]=V[j=1,k=ks+1], B[1,0]=V[j=2,k=ks], B[1,1]=V[j=2,k=ks+1]
           Output C sub-block: C[i, ks+k'] for i in {0,1}, k' in {0,1}
           Strassen:
             M1 = (A00+A11)*(B00+B11)
             M2 = (A10+A11)*B00
             M3 = A00*(B01-B11)
             M4 = A11*(B10-B00)
             M5 = (A00+A01)*B11
             M6 = (A10-A00)*(B00+B01)
             M7 = (A01-A10)*(B10+B11)
             C00 = M1+M4-M5+M7
             C01 = M3+M5
             C10 = M2+M4
             C11 = M1-M2+M3+M6
           Each M is one rank-1 term.
        """
        global col
        # Map local indices to global U/V/W rows
        # U row for (i, j_local): i*m + (1 + j_local)  where j_local 0->j=1, 1->j=2
        # V row for (j_local, k_local): (1+j_local)*p + (k_start + k_local)
        # W row for (k_local, i): (k_start + k_local)*n + i

        def U_row(i, jl):
            return i * m + (1 + jl)

        def V_row(jl, kl):
            return (1 + jl) * p + (k_start + kl)

        def W_row(kl, i):
            return (k_start + kl) * n + i

        # M1 = (A00+A11)*(B00+B11)
        # A00 = U(i=0,jl=0), A11 = U(i=1,jl=1)
        # B00 = V(jl=0,kl=0), B11 = V(jl=1,kl=1)
        # C contribution: C00 += M1, C11 += M1
        # This is NOT a single rank-1 term; it's a product of sums.
        # In tensor decomposition, each "multiplication" M corresponds to one column
        # where U, V have the sum of appropriate entries, and W has 1 for the output.
        # But a sum in U and sum in V means the column has multiple nonzeros in U and V.
        # That's fine for tensor rank decomposition.

        # Let me define each Strassen multiplication as one column:
        # For M1: U has 1 at rows U_row(0,0) and U_row(1,1); V has 1 at rows V_row(0,0) and V_row(1,1)
        #         W has 1 at rows W_row(0,0) and W_row(1,1)
        # Actually no - in the tensor model, the decomposition is:
        #   sum_c U[a,c] * V[b,c] * W[d,c] = T[a,b,d]
        # Each column c contributes U[a,c]*V[b,c]*W[d,c] to T[a,b,d].
        # For a product like (A00+A11)*(B00+B11), we need:
        #   (A00*A00 + A00*B11 + A11*B00 + A11*B11)
        # That's 4 terms, not 1! Strassen works over matrices where you can add matrices,
        # but in tensor decomposition each "multiplication" is a scalar product.
        #
        # I was confused. Strassen's algorithm reduces the NUMBER OF SCALAR MULTIPLICATIONS
        # when multiplying matrices, but each scalar multiplication in Strassen corresponds
        # to a product of two linear combinations of input entries. In the tensor rank
        # decomposition, each such product IS one rank-1 term (one column in U,V,W).
        # The "sum" happens in the pre/post multiplication (linear combinations in U and V).
        #
        # So for M1 = (A00+A11)*(B00+B11):
        #   One column where U has 1 at A00 and A11 positions, V has 1 at B00 and B11 positions,
        #   and W has 1 at the output positions.
        #   This gives contribution: (A00+A11)*(B00+B11) * (output basis)
        #   Which equals A00*B00 + A00*B11 + A11*B00 + A11*B11 times the output basis.
        #   Yes! That's correct. Each Strassen M_i is ONE rank-1 term.

        # M1: (A00+A11)*(B00+B11) -> contributes to C00 and C11
        U[U_row(0, 0)][col] += 1
        U[U_row(1, 1)][col] += 1
        V[V_row(0, 0)][col] += 1
        V[V_row(1, 1)][col] += 1
        W[W_row(0, 0)][col] += 1  # C00
        W[W_row(1, 1)][col] += 1  # C11
        col += 1

        # M2: (A10+A11)*B00 -> contributes to C10
        U[U_row(1, 0)][col] += 1
        U[U_row(1, 1)][col] += 1
        V[V_row(0, 0)][col] += 1
        W[W_row(0, 1)][col] += 1  # C10
        col += 1

        # M3: A00*(B01-B11) -> contributes to C01
        U[U_row(0, 0)][col] += 1
        V[V_row(0, 1)][col] += 1
        V[V_row(1, 1)][col] -= 1
        W[W_row(1, 0)][col] += 1  # C01
        col += 1

        # M4: A11*(B10-B00) -> contributes to C10 and C00
        U[U_row(1, 1)][col] += 1
        V[V_row(1, 0)][col] += 1
        V[V_row(0, 0)][col] -= 1
        W[W_row(0, 1)][col] += 1  # C10
        W[W_row(0, 0)][col] += 1  # C00
        col += 1

        # M5: (A00+A01)*B11 -> contributes to C11 and C01
        U[U_row(0, 0)][col] += 1
        U[U_row(0, 1)][col] += 1
        V[V_row(1, 1)][col] += 1
        W[W_row(1, 1)][col] += 1  # C11
        W[W_row(1, 0)][col] += 1  # C01
        col += 1

        # M6: (A10-A00)*(B00+B01) -> contributes to C11
        U[U_row(1, 0)][col] += 1
        U[U_row(0, 0)][col] -= 1
        V[V_row(0, 0)][col] += 1
        V[V_row(0, 1)][col] += 1
        W[W_row(1, 1)][col] += 1  # C11
        col += 1

        # M7: (A01-A10)*(B10+B11) -> contributes to C00
        U[U_row(0, 1)][col] += 1
        U[U_row(1, 0)][col] -= 1
        V[V_row(1, 0)][col] += 1
        V[V_row(1, 1)][col] += 1
        W[W_row(0, 0)][col] += 1  # C00
        col += 1

    # Apply Strassen to four 2x2x2 blocks
    strassen_block(0, 1)   # k=0,1; j=1,2
    strassen_block(2, 1)   # k=2,3; j=1,2
    strassen_block(4, 1)   # k=4,5; j=1,2
    strassen_block(6, 1)   # k=6,7; j=1,2

    # --- Last k-block: k=8, width 1: schoolbook 2x2x1 = 4 terms ---
    for i in range(2):
        for jl in range(2):
            U[U_row(i, jl)][col] = 1
            V[V_row(jl, 0)][col] = 1  # k_local=0, k_start=8
            W[W_row(0, i)][col] = 1   # k=8
            col += 1

    # Verify col == 50
    assert col == 50, f"Expected 50 columns, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
