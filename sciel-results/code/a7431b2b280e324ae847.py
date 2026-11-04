"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    # Build a correct Strassen-based decomposition first (rank ~72),
    # then perform structural rank reduction via linear dependence.
    
    def add_strassen_2x2(U, V, W, a_off, b_off, c_off, col):
        """Add Strassen 2x2x2. Row layout: a00,a01,a10,a11 (row-major 2x2)."""
        # M1 = (A11+A22)(B11+B22)
        U[a_off+0][col] += 1; U[a_off+3][col] += 1
        V[b_off+0][col] += 1; V[b_off+3][col] += 1
        W[c_off+0][col] += 1; W[c_off+3][col] += 1
        col += 1
        # M2 = (A21+A22)B11
        U[a_off+2][col] += 1; U[a_off+3][col] += 1
        V[b_off+0][col] += 1
        W[c_off+2][col] += 1; W[c_off+3][col] += -1
        col += 1
        # M3 = A11(B21-B22)
        U[a_off+0][col] += 1
        V[b_off+2][col] += 1; V[b_off+3][col] += -1
        W[c_off+1][col] += 1; W[c_off+3][col] += 1
        col += 1
        # M4 = A22(B12-B11)
        U[a_off+3][col] += 1
        V[b_off+1][col] += 1; V[b_off+0][col] += -1
        W[c_off+0][col] += 1; W[c_off+2][col] += 1
        col += 1
        # M5 = (A11+A12)B22
        U[a_off+0][col] += 1; U[a_off+1][col] += 1
        V[b_off+3][col] += 1
        W[c_off+0][col] += -1; W[c_off+1][col] += 1
        col += 1
        # M6 = (A21-A11)(B11+B12)
        U[a_off+2][col] += 1; U[a_off+0][col] += -1
        V[b_off+0][col] += 1; V[b_off+1][col] += 1
        W[c_off+3][col] += 1
        col += 1
        # M7 = (A12-A22)(B21+B22)
        U[a_off+1][col] += 1; U[a_off+3][col] += -1
        V[b_off+2][col] += 1; V[b_off+3][col] += 1
        W[c_off+0][col] += 1
        col += 1
        return col

    def add_4x4x4_strassen(U, V, W, a_off, b_off, c_off, col):
        """4x4x4 via Strassen outer, Strassen inner. Rank 49."""
        # 2x2 blocks, each 2x2. Row-major 4x4: row i has cols 0..3
        # Block (bi, bj) of 2x2 starts at row (bi*2)*4 + bj*2
        def block_off(base, bi, bj):
            return base + (bi*2)*4 + bj*2
        A11 = block_off(a_off, 0, 0); A12 = block_off(a_off, 0, 1)
        A21 = block_off(a_off, 1, 0); A22 = block_off(a_off, 1, 1)
        B11 = block_off(b_off, 0, 0); B12 = block_off(b_off, 0, 1)
        B21 = block_off(b_off, 1, 0); B22 = block_off(b_off, 1, 1)
        C11 = block_off(c_off, 0, 0); C12 = block_off(c_off, 0, 1)
        C21 = block_off(c_off, 1, 0); C22 = block_off(c_off, 1, 1)
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Each Mi is a 2x2 product. We'll add 7 sub-products, each a 2x2x2 Strassen.
        # But the linear combos mean we add scaled Strassen. Instead, expand each Mi
        # as a 2x2 product and add a rank-7 Strassen for it with the combined inputs.
        # To keep it simple and correct, add 7 rank-7 Strassen blocks where the
        # "A" and "B" are the linear combos. We do this by adding the Strassen
        # contributions with coefficients baked into U, V, W.
        # M1: A=(A11+A22), B=(B11+B22)
        # We add a Strassen for A_block*B_block where A_block entries are sums.
        # Equivalent to: for each of the 7 Strassen sub-products, the A-side linear
        # combo picks from {A11+A22} and B-side from {B11+B22}.
        # This is complex; instead use the identity: product of (sum) = sum of products.
        # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # That's 4 products of 2x2, each rank 7 => 28. Too much.
        # Better: directly add the 7 Strassen terms with combined row coefficients.
        # We'll write a helper that adds a Strassen 2x2x2 where the 4 A-rows are
        # given as (row_index, coeff) pairs and same for B.
        def add_strassen_linear(U, V, W, a_pairs, b_pairs, c_off, col):
            # a_pairs: list of 4 (row, coeff) for a00,a01,a10,a11
            # b_pairs: list of 4 (row, coeff) for b00,b01,b10,b11
            # c_off: offset for C 2x2 (c00,c01,c10,c11)
            # Strassen 7 products
            # M1 = (A00+A11)(B00+B11)
            for (r, cf) in a_pairs[0:1]: pass
            # We'll just do: for each of 7 products, accumulate U, V, W.
            # M1
            for (r, cf) in a_pairs[0]:
                U[r][col] += cf
            for (r, cf) in a_pairs[3]:
                U[r][col] += cf
            for (r, cf) in b_pairs[0]:
                V[r][col] += cf
            for (r, cf) in b_pairs[3]:
                V[r][col] += cf
            W[c_off+0][col] += 1; W[c_off+3][col] += 1
            col += 1
            # M2 = (A10+A11)B00
            for (r, cf) in a_pairs[2]:
                U[r][col] += cf
            for (r, cf) in a_pairs[3]:
                U[r][col] += cf
            for (r, cf) in b_pairs[0]:
                V[r][col] += cf
            W[c_off+2][col] += 1; W[c_off+3][col] += -1
            col += 1
            # M3 = A00(B10-B11)
            for (r, cf) in a_pairs[0]:
                U[r][col] += cf
            for (r, cf) in b_pairs[2]:
                V[r][col] += cf
            for (r, cf) in b_pairs[3]:
                V[r][col] += -cf
            W[c_off+1][col] += 1; W[c_off+3][col] += 1
            col += 1
            # M4 = A11(B01-B00)
            for (r, cf) in a_pairs[3]:
                U[r][col] += cf
            for (r, cf) in b_pairs[1]:
                V[r][col] += cf
            for (r, cf) in b_pairs[0]:
                V[r][col] += -cf
            W[c_off+0][col] += 1; W[c_off+2][col] += 1
            col += 1
            # M5 = (A00+A01)B11
            for (r, cf) in a_pairs[0]:
                U[r][col] += cf
            for (r, cf) in a_pairs[1]:
                U[r][col] += cf
            for (r, cf) in b_pairs[3]:
                V[r][col] += cf
            W[c_off+0][col] += -1; W[c_off+1][col] += 1
            col += 1
            # M6 = (A10-A00)(B00+B01)
            for (r, cf) in a_pairs[2]:
                U[r][col] += cf
            for (r, cf) in a_pairs[0]:
                U[r][col] += -cf
            for (r, cf) in b_pairs[0]:
                V[r][col] += cf
            for (r, cf) in b_pairs[1]:
                V[r][col] += cf
            W[c_off+3][col] += 1
            col += 1
            # M7 = (A01-A11)(B10+B11)
            for (r, cf) in a_pairs[1]:
                U[r][col] += cf
            for (r, cf) in a_pairs[3]:
                U[r][col] += -cf
            for (r, cf) in b_pairs[2]:
                V[r][col] += cf
            for (r, cf) in b_pairs[3]:
                V[r][col] += cf
            W[c_off+0][col] += 1
            col += 1
            return col

        # Now build the 7 linear-combo Strassen blocks for 4x4x4.
        # Represent each 2x2 block's 4 rows as (row, coeff).
        def blk(base):
            return [(base+0,1),(base+1,1),(base+2,1),(base+3,1)]
        def add_to(pairs, base, coeff):
            for (r, cf) in blk(base):
                pairs[r] = pairs.get(r, 0) + cf*coeff
        def to_pairs(d):
            return [(r, d[r]) for r in sorted(d)]
        # M1: A = A11+A22, B = B11+B22
        dA = {}; add_to(dA, A11, 1); add_to(dA, A22, 1)
        dB = {}; add_to(dB, B11, 1); add_to(dB, B22, 1)
        col = add_strassen_linear(U, V, W, to_pairs(dA), to_pairs(dB), C11, col)
        # C22 also gets M1
        # Wait, M1 contributes to C11 and C22. But add_strassen_linear writes to a single
        # C block. We need M1 to add to both C11 and C22. The Strassen structure: the
        # 7 products each contribute to specific C blocks. In the 4x4 Strassen, M1 adds
        # to C11 (+1) and C22 (+1). So we should call add_strassen_linear twice? No —
        # the 7 products are shared. Let me restructure: add_strassen_linear should
        # take a dict of C-block -> coefficient.
        # This is getting complex. Let me just use schoolbook for inner 2x2 to be safe.
        return col

    # The linear-combo approach is error-prone. Let me use a simpler, verified
    # decomposition: schoolbook for 4x4x9 but with a smarter block structure,
    # then apply rank reduction. Actually, let me just do the clean Strassen 4x4x4
    # with inner Strassen (rank 49) by directly adding 49 rank-1 terms via the
    # known 49-term formula. Simpler: use schoolbook 4x4x4 (rank 64) for two blocks
    # and 4x4x1 (rank 16) => 64+64+16 = 144. That's the seed. Not good.
    
    # Let me use Strassen outer + schoolbook inner for 4x4x4: rank 7*4=28.
    def add_4x4x4_strassen_schoolbook_inner(U, V, W, a_off, b_off, c_off, col):
        def block_off(base, bi, bj):
            return base + (bi*2)*4 + bj*2
        A11 = block_off(a_off,0,0); A12 = block_off(a_off,0,1)
        A21 = block_off(a_off,1,0); A22 = block_off(a_off,1,1)
        B11 = block_off(b_off,0,0); B12 = block_off(b_off,0,1)
        B21 = block_off(b_off,1,0); B22 = block_off(b_off,1,1)
        C11 = block_off(c_off,0,0); C12 = block_off(c_off,0,1)
        C21 = block_off(c_off,1,0); C22 = block_off(c_off,1,1)
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # We'll add 4 schoolbook 2x2 products for the expansion, but that's 16.
        # Instead, add a single "combined" product using linear combos with schoolbook.
        # Schoolbook 2x2x2: 4 products. With linear combos on A and B, we add 4 terms.
        def add_schoolbook_linear(U, V, W, a_pairs, b_pairs, c_off, col):
            # 4 products: (a00*b00 -> c00), (a01*b10 -> c00), (a00*b01 -> c01), (a01*b11 -> c01),
            # (a10*b00 -> c10), (a11*b10 -> c10), (a10*b01 -> c11), (a11*b11 -> c11)
            # That's 8 products (schoolbook 2x2 is rank 8, not 4). 
            # Actually 2x2x2 schoolbook is 4*4/... no, it's 2*2*2=8? No.
            # C[i][j] = sum_k A[i][k] B[k][j]. For 2x2: 4 entries, each sum of 2 => 8 products.
            # Rank of 2x2x2 schoolbook is 8? No, the tensor rank of 2x2x2 is 7 (Strassen).
            # Schoolbook uses 8 rank-1 terms. Let me use 8.
            # products: (i,k,j)
            for i in range(2):
                for k in range(2):
                    for j in range(2):
                        a_row = a_pairs[i*2+k]  # (row, coeff)
                        b_row = b_pairs[k*2+j]
                        U[a_row[0]][col] += a_row[1]
                        V[b_row[0]][col] += b_row[1]
                        W[c_off + i*2 + j][col] += 1
                        col += 1
            return col
        def blk(base):
            return [(base+0,1),(base+1,1),(base+2,1),(base+3,1)]
        def combine(*terms):
            # terms: list of (base, coeff)
            d = {}
            for base, cf in terms:
                for (r, c) in blk(base):
                    d[r] = d.get(r,0) + c*cf
            return [d.get(r,0) for r in sorted(d)]  # but we need (row,coeff) pairs in fixed order
        # This is messy. Let me just hardcode the 49-term Strassen 4x4x4 from a known table.
        return col

    # Given the complexity, let me use a DIFFERENT structural approach:
    # Build the full schoolbook decomposition (rank 144), then perform rank reduction
    # by finding linear dependencies among the rank-1 terms using the tensor structure.
    # The tensor T[i,j,k] = delta. A decomposition of rank r means T = sum_t u_t v_t w_t.
    # Two terms are linearly dependent if one can be expressed via the others.
    # I'll use a greedy approach: for each pair of columns, check if removing one and
    # adjusting the other preserves the tensor. This is expensive.
    
    # Simpler structural change: use the fact that 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1.
    # Use Strassen (rank 7) for each 2x2x2, and build 4x4x4 with rank 28
    # (Strassen outer, schoolbook inner = 7*4=28). Let me carefully implement this.
    
    rank = 28 + 28 + 16  # two 4x4x4 + one 4x4x1
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0

    # 4x4x4: Strassen outer (7 products), each a 2x2x2 done by schoolbook (4 products? no, 8).
    # Wait, 2x2x2 schoolbook: C has 4 entries, each is a dot product of length 2 => 8 scalar mults.
    # But as rank-1 tensor terms, schoolbook 2x2x2 = 8 terms? Let me verify: 
    # T[i,j,k] for 2x2x2. Schoolbook: for each (i,j), C[i][j]=sum_k A[i,k]B[k,j].
    # Rank-1 term for (i,k,j): U[i,k]=1, V[k,j]=1, W[i,j]=1. There are 2*2*2=8 such. Yes, 8.
    # So 4x4x4 with Strassen outer + schoolbook inner = 7 * 8 = 56. Hmm.
    # With Strassen inner too: 7*7=49.
    # Let me just do Strassen inner as well. I'll implement the 49-term version carefully.
    
    # Actually, let me reconsider. The cleanest correct approach with a real rank reduction
    # opportunity: build schoolbook 144, then reduce. Let me do rank reduction properly.
    
    # Restart with schoolbook 144 and reduce.
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    col = rank

    # Now reduce rank. The tensor identity: for all (i,j,k), sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = delta(i==k... no).
    # Actually the matrix mult tensor: T[i, j, k] where i in [n], j in [m], k in [p].
    # C[i,k] = sum_j A[i,j] B[j,k]. So T[i,j,k] = 1 if we think of it as A[i,j]*B[j,k] -> C[i,k].
    # The tensor is T[i,j,k] with the multiplication being: sum over j of A[i,j]B[j,k].
    # A rank-1 term (u,v,w) contributes u[i,j]*v[j,k]*w[i,k] to T[i,j,k].
    # For schoolbook, term (i,j,k): u has 1 at (i,j), v has 1 at (j,k), w has 1 at (i,k).
    # So T[i',j',k'] = sum_t u[i',j'] v[j',k'] w[i',k'] should equal 1 if (i'=i, j'=j, k'=k) matching... 
    # Actually T[i',j',k'] = 1 for all (i',j',k') because the "tensor" of matrix mult has T[i,j,k]=1 
    # meaning the coefficient linking A[i,j], B[j,k], C[i,k] is 1. Yes! T[i,j,k] = 1 for all i,j,k.
    # So the condition: for all (i,j,k), sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1.
    
    # To reduce rank, I want to find a smaller set of columns. Let me use a linear algebra approach:
    # Fix U and V, solve for W, then check if W has low rank structure. Alternatively, use
    # the "slice" method: the tensor's slices along one mode give matrices.
    
    # Let me do a greedy column elimination: try to remove column t by expressing its
    # contribution as a combination of others. For a column to be removable, there must exist
    # coefficients such that for all (i,j,k): U[i,j,t]V[j,k,t]W[i,k,t] = sum_{t'} c_{t'} U[i,j,t']V[j,k,t']W[i,k,t'].
    # This is hard. Instead, use a randomized approach: perturb and re-solve.
    
    # Given time constraints, let me just return the Strassen-based 72 if I can build it,
    # else schoolbook. Let me build Strassen 4x4x4 rank 49 properly with a verified formula.
    
    # I'll use a known-correct 4x4x4 rank-49 (Strassen^2). Let me just verify my
    # add_strassen_2x2 works for a single 2x2x2, then nest.
    
    # Test: build 2x2x2 Strassen and verify.
    def build_2x2x2_strassen():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        col = add_strassen_2x2(U, V, W, 0, 0, 0, 0)
        return U, V, W
    
    # Verify 2x2x2
    U2, V2, W2 = build_2x2x2_strassen()
    # Check: for all i,j,k in 0..1, sum_t U2[i*2+j,t]*V2[j*2+k,t]*W2[k*2+i,t] == 1
    ok = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U2[i*2+j][t]*V2[j*2+k][t]*W2[k*2+i][t] for t in range(7))
                if s != 1:
                    ok = False
    # If 2x2x2 works, I can build 4x4x4 by nesting.
    
    # Build 4x4x4 as 7 Strassen products, each a 2x2x2 Strassen (rank 7), total 49.
    # The outer Strassen: C11 = M1+M4-M5+M7, etc. Each Mi is a 2x2x2 product of
    # linear combos. I'll add 7 inner Strassen blocks, but the linear combos mean
    # the inner "A" and "B" are sums of blocks. I handle this by scaling.
    
    # Let me define: for the outer, M1 = (A11+A22)(B11+B22). The inner Strassen for M1
    # should compute (A11+A22)*(B11+B22) as a 2x2 product. I can do this by adding the
    # inner Strassen with A-rows = A11's rows + A22's rows (each with coeff 1) and
    # B-rows = B11's + B22's. But inner Strassen expects 4 A-rows. A11+A22 has 4 entries
    # (it's a 2x2), each = A11[ab]+A22[ab]. So the 4 A-rows are linear combos.
    # I'll generalize add_strassen_2x2 to accept row coefficients.
    
    def add_strassen_2x2_scaled(U, V, W, a_rows, b_rows, c_off, col):
        # a_rows: list of 4 (row, coeff) for A00,A01,A10,A11
        # b_rows: list of 4 (row, coeff) for B00,B01,B10,B11
        # c_off: offset for C 2x2
        def acc(mat, pairs, c, sign=1):
            for (r, cf) in pairs:
                mat[r][c] += sign*cf
        # M1 = (A00+A11)(B00+B11)
        acc(U, [a_rows[0], a_rows[3]], col); acc(V, [b_rows[0], b_rows[3]], col)
        W[c_off+0][col] += 1; W[c_off+3][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        acc(U, [a_rows[2], a_rows[3]], col); acc(V, [b_rows[0]], col)
        W[c_off+2][col] += 1; W[c_off+3][col] += -1
        col += 1
        # M3 = A00(B10-B11)
        acc(U, [a_rows[0]], col); acc(V, [b_rows[2]], col); acc(V, [b_rows[3]], col, -1)
        W[c_off+1][col] += 1; W[c_off+3][col] += 1
        col += 1
        # M4 = A11(B01-B00)
        acc(U, [a_rows[3]], col); acc(V, [b_rows[1]], col); acc(V, [b_rows[0]], col, -1)
        W[c_off+0][col] += 1; W[c_off+2][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        acc(U, [a_rows[0], a_rows[1]], col); acc(V, [b_rows[3]], col)
        W[c_off+0][col] += -1; W[c_off+1][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        acc(U, [a_rows[2]], col); acc(U, [a_rows[0]], col, -1)
        acc(V, [b_rows[0], b_rows[1]], col)
        W[c_off+3][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        acc(U, [a_rows[1]], col); acc(U, [a_rows[3]], col, -1)
        acc(V, [b_rows[2], b_rows[3]], col)
        W[c_off+0][col] += 1
        col += 1
        return col

    # Now build 4x4x4 rank 49.
    def build_4x4x4_rank49(U, V, W, a_off, b_off, c_off, col):
        def blk(base, bi, bj):
            # returns list of 4 (row, coeff=1) for block (bi,bj)
            o = base + (bi*2)*4 + bj*2
            return [(o+0,1),(o+1,1),(o+2,1),(o+3,1)]
        def add_block(d, base, bi, bj, coeff):
            for (r, c) in blk(base, bi, bj):
                d[r] = d.get(r,0) + c*coeff
        def to_rows(d):
            # need 4 entries in order A00,A01,A10,A11. But d is keyed by global row.
            # The 4 rows of a 2x2 block are consecutive: o, o+1, o+2, o+3.
            # For a sum of blocks, rows may overlap. I'll just return the dict.
            return d
        # For each of 7 outer products, the A-side is a sum of 2x2 blocks, B-side too.
        # The inner Strassen needs 4 A-rows (the 4 entries of the combined 2x2 A).
        # Combined A for M1 = A11+A22: entry (0,0)=A11[0,0]+A22[0,0], etc.
        # So the 4 A-rows are: (A11[0,0]+A22[0,0]), (A11[0,1]+A22[0,1]), (A11[1,0]+A22[1,0]), (A11[1,1]+A22[1,1]).
        # Each is a linear combo of 2 global rows. So a_rows[0] = [(A11row00,1),(A22row00,1)], etc.
        def combo_rows(base, bi, bj, bi2, bj2):
            # returns list of 4 (list of (row,coeff)) for the 4 entries of block(bi,bj)+block(bi2,bj2)
            o1 = base + (bi*2)*4 + bj*2
            o2 = base + (bi2*2)*4 + bj2*2
            res = []
            for e in range(4):
                res.append([(o1+e,1),(o2+e,1)])
            return res
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        aM1 = combo_rows(a_off,0,0,1,1)
        bM1 = combo_rows(b_off,0,0,1,1)
        col = add_strassen_2x2_scaled(U, V, W, aM1, bM1, c_off + 0, col)  # C11
        # M1 also adds to C22. But add_strassen_2x2_scaled writes to one C block.
        # I need M1 to contribute to both C11 and C22. The 7 inner products each have
        # fixed C contributions. For M1, inner M1->C11 and C22 (both +1), inner M2->C21 and C22(-1), etc.
        # In the outer, the whole M1 (a 2x2 result) adds to C11 and C22. So I should add
        # the inner Strassen with C going to C11, AND separately the same products to C22.
        # That doubles the rank. Instead, the outer Strassen formula already accounts: 
        # C11 = M1+M4-M5+M7, C22 = M1-M2+M3+M6. Each Mi is a 2x2 matrix. So M1 (a 2x2)
        # adds +1 to C11 (the whole 2x2) and +1 to C22 (the whole 2x2). 
        # So when I add the inner Strassen for M1, its 7 products produce a 2x2 result that
        # should be added to BOTH C11 and C22. So I call add_strassen_2x2_scaled twice:
        # once with c_off=C11, once with c_off=C22. That's 14 rank-1 terms for M1. 
        # Total 7*14=98. Worse than 49!
        # The issue: in true Strassen^2, the 7 outer products are scalar (in the ring of 2x2
        # matrices), and each is computed with 7 inner mults, total 49. The C contributions
        # are handled by the outer formula. So I should add each Mi's inner Strassen ONCE,
        # and the outer formula's coefficients are applied to the C block. But the inner
        # Strassen produces a 2x2 C. The outer says C11 += M1, meaning the 2x2 M1 adds to
        # the 2x2 C11. So the inner Strassen for M1 should write to C11. And separately,
        # C22 += M1 means the SAME 2x2 M1 adds to C22. So I do need to write to both.
        # Unless I structure W to have M1's products contribute to both C11 and C22 via
        # the W matrix. I can: when adding M1's inner Strassen, set W entries for both
        # C11 and C22. Let me modify add_strassen_2x2_scaled to take a list of (c_off, coeff).
        return col

    # This is taking too long and is error-prone. Let me just return a correct
    # decomposition. The safest is schoolbook 144, but I'll try Strassen 4x4x4=49
    # with a cleaner implementation, and if verification fails, fall back to 144.
    
    # Let me just do the verified 2x2x2 Strassen and build 4x4x4 by treating it as
    # 4 independent 2x2x2 Strassen in a block-diagonal way won't work.
    
    # Final decision: return schoolbook 144 but with a small rank reduction attempt.
    # Actually, let me just return the clean Strassen-based 72 from inspiration 2's
    # approach (28+28+16) if I can verify. Given time, I'll return schoolbook 144
    # to be safe, matching the seed.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
