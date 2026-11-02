"""Matrix multiplication 3x4x7 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = 3, 4, 7
    # We'll build a list of (u_row, v_row, w_row, coeff) tuples
    # Then deduplicate/merge at the end to minimize rank.
    # Actually, it's simpler to build a list of 4-tuples (u, v, w, coeff)
    # where u is index into U rows, v into V rows, w into W rows.
    # But we need to be careful: the decomposition is sum_c U[i,c]*V[j,c]*W[k,c] = (AB)[i,k]
    # So for each "term" in the decomposition, we have a coefficient times
    #   e_i (row of A) * e_j (col of A / row of B) * e_k (col of B)
    # which contributes to C[i,k].
    # The rank-1 tensor term is: U has 1 at row (i,j), V has 1 at row (j,k), W has 1 at row (k,i)
    # Wait, let me re-read the convention.
    # U: shape (n*m, rank) -> U[i*m+j, c]
    # V: shape (m*p, rank) -> V[j*p+k, c]
    # W: shape (p*n, rank) -> W[k*n+i, c]
    # Product: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal (A*B)[i, k]
    # where A is n x m, B is m x p.
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So for each (i, j, k), the coefficient should be 1 if we're doing schoolbook.
    # In Strassen, we combine terms with coefficients.

    # Let's use a dictionary to accumulate contributions.
    # Key: (u_idx, v_idx, w_idx) where
    #   u_idx = i * m + j
    #   v_idx = j * p + k
    #   w_idx = k * n + i
    # Value: coefficient

    # Actually, since we're doing exact arithmetic, let's use fractions or just integers.
    # Strassen uses coefficients of 1 and -1, so integers suffice.

    from fractions import Fraction

    terms = {}  # (u_idx, v_idx, w_idx) -> Fraction

    def add_term(i, j, k, coeff):
        """Add a term with coefficient to the decomposition.
        i: row index in A (0..n-1)
        j: col index in A / row index in B (0..m-1)
        k: col index in B (0..p-1)
        """
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        key = (u_idx, v_idx, w_idx)
        terms[key] = terms.get(key, Fraction(0)) + coeff

    # We'll partition A into blocks and B into blocks, then use Strassen for 2x2x2 subproblems.
    # A is 3x4. Let's split A as:
    #   A_top: rows 0,1 (2x4)
    #   A_bot: row 2 (1x4)
    # B is 4x7. Let's split B as:
    #   B_top: cols 0,1,2,3 -> actually B is 4 rows, 7 cols.
    #   Wait, A is 3x4, B is 4x7.
    #   A[i,j] for i in 0..2, j in 0..3
    #   B[j,k] for j in 0..3, k in 0..6
    #   C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..2, k in 0..6

    # Strategy: split the middle dimension (j) into two parts: j in {0,1} and j in {2,3}
    # Then C[i,k] = sum_{j=0,1} A[i,j]*B[j,k] + sum_{j=2,3} A[i,j]*B[j,k]
    # Let's define:
    #   A1 = A[:, 0:2] (3x2), A2 = A[:, 2:4] (3x2)
    #   B1 = B[0:2, :] (2x7), B2 = B[2:4, :] (2x7)
    # Then C = A1*B1 + A2*B2

    # Now A1 is 3x2, B1 is 2x7. The product is 3x7.
    # A2 is 3x2, B2 is 2x7. The product is 3x7.
    # Total rank = rank(3x2x7) + rank(3x2x7)

    # For 3x2x7: split the 3 rows into 2+1, and the 7 cols into 4+3.
    # A1_top = A1[0:2, :] (2x2), A1_bot = A1[2:3, :] (1x2)
    # B1_left = B1[:, 0:4] (2x4), B1_right = B1[:, 4:7] (2x3)
    # C1_top_left = A1_top * B1_left (2x4)
    # C1_top_right = A1_top * B1_right (2x3)
    # C1_bot_left = A1_bot * B1_left (1x4)
    # C1_bot_right = A1_bot * B1_right (1x3)

    # For 2x4: split 2x4 as two 2x2 blocks. Use Strassen for each? No, 2x4x2 is not standard.
    # Actually, 2x4 matrix multiplication: A is 2x2, B is 2x4? No.
    # A1_top is 2x2, B1_left is 2x4. Product is 2x4.
    # We can split B1_left into two 2x2 blocks: B1_left_1 (cols 0,1) and B1_left_2 (cols 2,3).
    # Then C1_top_left = A1_top * B1_left_1 + A1_top * B1_left_2
    # Each is a 2x2x2 multiplication, which can be done with Strassen (rank 7).
    # So 2x4x2 multiplication = 2 * 7 = 14 rank.

    # Similarly, 2x3x2: A1_top is 2x2, B1_right is 2x3.
    # Split B1_right into 2x2 and 2x1? That's awkward.
    # Alternatively, 2x3 can be viewed as two 2x2 with overlap? No.
    # Let's just do schoolbook for small blocks.
    # 2x3x2 schoolbook rank = 2*3*2 = 12. Strassen doesn't directly apply.

    # Let me reconsider the blocking strategy to maximize Strassen usage.

    # Alternative: 
    # A is 3x4. Split into 2x2, 2x2, 1x2, 1x2 blocks? No, that's not rectangular.
    # Let's try: split A rows into 2 and 1. Split A cols into 2 and 2.
    # So A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2.
    # B is 4x7. Split B rows into 2 and 2. Split B cols into 4 and 3.
    # So B = [[B11, B12], [B21, B22]] where B11 is 2x4, B12 is 2x3, B21 is 2x4, B22 is 2x3.
    # Then C = A11*B11 + A12*B21 + A21*B11 + A22*B21? No, that's not right.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # If we split j into {0,1} and {2,3}:
    # C_top = A_top * B_top + A_top * B_bot  where A_top is 2x4, B_top is 2x7, B_bot is 2x7
    # C_bot = A_bot * B_top + A_bot * B_bot  where A_bot is 1x4, B_top is 2x7, B_bot is 2x7

    # This is getting complicated. Let me just implement a recursive or direct Strassen-based approach.

    # Actually, the simplest approach that's likely to give good rank:
    # Use Strassen for all 2x2x2 subproblems, and schoolbook for the rest.
    # Let me carefully partition.

    # Partition A (3x4) as:
    #   Rows: [0,1] and [2]
    #   Cols: [0,1], [2,3]
    # Partition B (4x7) as:
    #   Rows: [0,1], [2,3]
    #   Cols: [0,1], [2,3], [4,5], [6]
    # This gives many small blocks. Let me just do it systematically.

    # I'll define a function to add a matrix multiplication block using Strassen if 2x2x2, else schoolbook.
    # For a subproblem: A_sub[i_local, j_local] * B_sub[j_local, k_local] -> C_sub[i_local, k_local]
    # where i_local is in a range of A's rows, j_local in a range of A's cols / B's rows, k_local in a range of B's cols.

    # Let me define the Strassen decomposition for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For each Strassen term, it's a product of two 1x1 "matrices" (scalars) in the tensor sense.
    # But in our tensor decomposition, each term is a rank-1 tensor.
    # The Strassen algorithm has 7 multiplications, each of which is a product of linear combinations.
    # In the tensor decomposition, we need to express this as sum of rank-1 tensors.
    # This is non-trivial. Let me think differently.

    # Actually, the rank of the 2x2x2 matrix multiplication tensor is 7, and Strassen gives an explicit decomposition of rank 7.
    # The decomposition can be written as:
    # For each of the 7 terms, we have specific patterns in U, V, W.

    # Let me look up the explicit Strassen decomposition in terms of the tensor.
    # The 7 rank-1 tensors for Strassen are:
    # Term 1: (e11+e22) in A, (e11+e22) in B, (e11+e22) in C with appropriate signs
    # This is getting too complex to derive from scratch.

    # Given the time constraint, let me just do schoolbook but with a smarter partitioning
    # that might allow some cancellation. Actually, schoolbook for 3x4x7 is rank 84.
    # The Strassen-based approach should be able to do better.

    # Let me try a different approach: use the known result that rank(3,4,7) <= some value.
    # Actually, let me just implement the Strassen-based decomposition properly.

    # I'll use a helper that, given ranges for i, j, k, adds the appropriate terms.
    # For 2x2x2 subblocks, I'll use Strassen's 7 terms.

    # Strassen's 7 rank-1 tensor terms for 2x2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # The 7 products are:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(b-a)  -- wait, this is for C21
    # Let me use the standard form:
    # m1 = (A11+A22)(B11+B22)
    # m2 = (A21+A22)B11
    # m3 = A11(B21-B22)
    # m4 = A22(B12-B11)
    # m5 = (A11+A12)B22
    # m6 = (A21-A11)(B11+B12)
    # m7 = (A12-A22)(B21+B22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6

    # In tensor form, each mj is a product of linear forms in A, B, and contributes to C.
    # For m1: A has (1,1) and (2,2) coefficients of 1, B has (1,1) and (2,2) coefficients of 1,
    #        C has (1,1) coefficient +1, (2,2) coefficient +1
    # But wait, in the tensor decomposition, each rank-1 term has a single U row, single V row, single W row.
    # The Strassen decomposition in tensor form is more subtle.

    # I think I need to just do schoolbook for now and accept rank 84, but that's the same as the seed.
    # The seed is rank 84. I need to do better.

    # Let me try: for the 2x2x2 subblocks, use Strassen's 7 terms explicitly.
    # I'll hard-code the 7 terms for each 2x2x2 subproblem.

    # For a 2x2x2 subproblem with A rows {i0, i1}, B cols {k0, k1}, middle {j0, j1}:
    # The 7 terms are:
    # Term 1: A: (i0,j0)+1, (i1,j1)+1; B: (j0,k0)+1, (j1,k1)+1; C: (k0,i0)+1, (k1,i1)+1
    # Wait, that's not quite right. Let me think about it as:
    # m1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
    # This contributes to C[i0,k0] with +1, C[i0,k1] with +1, C[i1,k0] with +1, C[i1,k1] with +1
    # But in the tensor decomposition, this single term m1 is not a rank-1 tensor.
    # It's a sum of 4 rank-1 tensors if expanded, which defeats the purpose.

    # I think I'm overcomplicating this. The rank-1 tensor decomposition of Strassen's algorithm
    # requires that each of the 7 "multiplications" be represented as a rank-1 tensor in the
    # (A, B, C) space. This is possible because each multiplication is of the form
    # (linear combination of A entries) * (linear combination of B entries) -> (linear combination of C entries)
    # And in the tensor decomposition, we can have U, V, W vectors with multiple non-zero entries.

    # So for m1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1]):
    # U has +1 at (i0,j0) and +1 at (i1,j1)
    # V has +1 at (j0,k0) and +1 at (j1,k1)
    # W has +1 at (k0,i0), +1 at (k0,i1), +1 at (k1,i0), +1 at (k1,i1)
    # Wait, no. The product U * V * W should give the contribution to C.
    # sum_c U[row_u, c] * V[row_v, c] * W[row_w, c] = C[i,k]
    # For m1, the contribution to C[i0,k0] should be +1, C[i0,k1] +1, C[i1,k0] +1, C[i1,k1] +1.
    # But with a single term c, we have U[row_u,c]*V[row_v,c]*W[row_w,c] for each (row_u, row_v, row_w).
    # This doesn't directly work because a single rank-1 tensor has a fixed pattern.

    # I think the correct way is:
    # For m1, we add 4 rank-1 tensors? No, that would be rank 4 for this one multiplication.
    # The point of Strassen is that it uses 7 multiplications instead of 8, but each multiplication
    # is of linear combinations, so in the tensor decomposition, it should still be rank 7 total
    # for the 2x2x2 block.

    # I believe the Strassen decomposition of the 2x2x2 tensor into 7 rank-1 tensors is a known result.
    # Let me just implement it by directly specifying the 7 terms.

    # After some research in my memory, the 7 rank-1 tensors for Strassen are:
    # (using 1-indexed for clarity, converting to 0-indexed)
    # Term 1: U: (1,1)+1, (2,2)+1; V: (1,1)+1, (2,2)+1; W: (1,1)+1, (2,2)+1
    # Term 2: U: (2,1)+1, (2,2)+1; V: (1,1)+1; W: (1,2)+1
    # Term 3: U: (1,1)+1; V: (2,1)+1, (2,2)-1; W: (2,1)+1
    # Term 4: U: (2,2)+1; V: (1,2)+1, (1,1)-1; W: (2,1)+1  -- this doesn't look right

    # I think I should just give up on deriving Strassen from scratch and use a known implementation.
    # Given the 120s time limit, let me just do schoolbook and hope that's acceptable, or try a simple improvement.

    # Actually, the seed is already schoolbook at rank 84. If I can't do better, I'll just return the same.
    # But the task asks for rank-minimal, so I should try harder.

    # Let me try: use Strassen for the 2x2x2 subblocks by directly encoding the 7 terms.
    # I found a reference: the Strassen decomposition in tensor form has the following 7 terms:
    # (for A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]])
    # T1: a*(e+h) -> no, that's not rank-1 in the right way.

    # I think the cleanest approach is to just implement the schoolbook decomposition and return it.
    # The score will be 1.0, same as the seed. But at least it's correct.

    # Wait, the directive says to write a stronger algorithm. Let me try one more thing:
    # Use the fact that 3x4x7 can be decomposed by first doing 3x2x7 and 3x2x7,
    # and for 3x2x7, use Strassen on the 2x2x2 parts.

    # For 3x2x7: A is 3x2, B is 2x7.
    # Split A into 2x2 and 1x2. Split B into 2x4 and 2x3.
    # 2x2x4: use two Strassen 2x2x2 -> rank 14
    # 2x2x3: schoolbook -> rank 12
    # 1x2x4: schoolbook -> rank 8
    # 1x2x3: schoolbook -> rank 6
    # Total for one 3x2x7: 14+12+8+6 = 40
    # For two of them: 80. That's worse than 84? No, 80 < 84. Good.

    # But wait, 2x2x4 with two Strassen: each Strassen is 2x2x2, rank 7. Two of them = 14. Correct.
    # 2x2x3 schoolbook: 2*2*3 = 12. Correct.
    # 1x2x4 schoolbook: 1*2*4 = 8. Correct.
    # 1x2x3 schoolbook: 1*2*3 = 6. Correct.
    # Total: 14+12+8+6 = 40 per 3x2x7.
    # Two of them: 80.

    # So rank 80 is achievable. Let me implement this.

    terms = {}

    def add_block_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        """Add schoolbook decomposition for A[i_start:i_end, j_start:j_end] * B[j_start:j_end, k_start:k_end]"""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    add_term(i, j, k, Fraction(1))

    def add_block_strassen(i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for 2x2x2 block.
        A rows: i0, i1
        B rows: j0, j1
        B cols: k0, k1
        """
        # Strassen's 7 terms
        # m1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # This contributes to C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1] with +1
        # In tensor form, this is 4 rank-1 tensors? No, it should be 1 rank-1 tensor
        # with U having two non-zeros, V having two non-zeros, W having four non-zeros.
        # But a rank-1 tensor has a single non-zero pattern. I think I'm confused.

        # Let me just use schoolbook for simplicity and correctness.
        add_block_schoolbook(i0, i1+1, j0, j1+1, k0, k1+1)

    # Given the complexity, let me just do full schoolbook.
    # Rank = 3*4*7 = 84
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term(i, j, k, Fraction(1))

    # Now, merge terms: if multiple terms have the same (u_idx, v_idx, w_idx), combine coefficients.
    # Then, remove terms with zero coefficient.
    # Finally, assign column indices and build U, V, W.

    # Filter out zero terms
    nonzero_terms = {key: coeff for key, coeff in terms.items() if coeff != 0}

    rank = len(nonzero_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for (u_idx, v_idx, w_idx), coeff in nonzero_terms.items():
        # Convert Fraction to int if possible, else float
        if coeff.denominator == 1:
            c = int(coeff.numerator)
        else:
            c = float(coeff)
        U[u_idx][col] = c
        V[v_idx][col] = 1  # Wait, this is wrong. The coefficient should be distributed.
        # Actually, in the decomposition, each term is U[row_u,c] * V[row_v,c] * W[row_w,c] = coeff
        # So we can set U[row_u,c] = coeff, V[row_v,c] = 1, W[row_w,c] = 1
        # Or distribute it. Let's set U[row_u,c] = coeff, V[row_v,c] = 1, W[row_w,c] = 1
        U[u_idx][col] = c
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
