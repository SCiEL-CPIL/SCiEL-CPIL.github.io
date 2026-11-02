import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 5

    # Strassen 2x2x2 blocks placed at:
    blocks = [
        (0, 0, 0),  # i=0,1; j=0,1; k=0,1
        (0, 2, 0),  # i=0,1; j=2,3; k=0,1
        (0, 0, 2),  # i=0,1; j=0,1; k=2,3
        (0, 2, 2),  # i=0,1; j=2,3; k=2,3
    ]

    n_strassen = len(blocks) * 7  # 28
    n_school = n * m * p - len(blocks) * 8  # 43
    total_rank = n_strassen + n_school  # 71

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7

    # Mark covered triples
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True

    # Schoolbook for remaining
    schoolbook_cols = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    schoolbook_cols.append((i, j, k, col))
                    col += 1

    # --- Rank-merging heuristic ---
    # Try to find if any schoolbook term for i=2 can be expressed as a
    # linear combination of Strassen columns plus other schoolbook columns,
    # allowing us to eliminate it.
    #
    # Strategy: For each schoolbook term (i=2, j, k), check if the rank-1
    # tensor e_{2m+j} ⊗ e_{jp+k} ⊗ e_{kn+2} is in the span of existing
    # rank-1 tensors. If so, we can remove it and adjust other columns.
    #
    # More practically: try to find pairs of schoolbook terms that can be
    # merged using the Strassen structure.
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1

    # Verify base decomposition
    T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    if not np.allclose(T_decomp, target, atol=1e-10):
        # Fallback
        rank_s = n * m * p
        U_f = [[0]*rank_s for _ in range(n*m)]
        V_f = [[0]*rank_s for _ in range(m*p)]
        W_f = [[0]*rank_s for _ in range(p*n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][c2] = 1
                    V_f[j*p+k][c2] = 1
                    W_f[k*n+i][c2] = 1
                    c2 += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    # Attempt rank reduction: try to find a linear dependency among all rank-1 terms
    # that allows removing one schoolbook term.
    # 
    # For each schoolbook term s, check if removing it and adjusting the
    # Strassen columns can still reproduce the target.
    # This is equivalent to checking if the tensor T_s is in the span of
    # the other rank-1 tensors.
    
    # A more tractable approach: check if any two schoolbook terms can be
    # combined with Strassen terms to save one rank.
    # 
    # Let's try: for each pair of schoolbook terms (a, b) both with i=2,
    # check if we can find coefficients alpha, beta such that
    # alpha * T_a + beta * T_b can be absorbed into the Strassen part.
    
    # Actually, let's try a simpler but effective approach:
    # Check if the i=2 schoolbook terms have linear dependencies with
    # the Strassen terms when projected onto certain subspaces.
    
    # For the i=2 terms, the U factor is e_{2m+j}, which is orthogonal to
    # all Strassen U entries (which only have i=0 or i=1). So we cannot
    # directly merge i=2 terms with Strassen terms in U.
    #
    # However, we CAN look for dependencies among the i=2 schoolbook terms
    # themselves, or between i=2 terms and the overall tensor structure.
    
    # Since U entries for i=2 are standard basis vectors that don't appear
    # in Strassen, the i=2 schoolbook terms are linearly independent in the
    # U-factor space. So we cannot reduce below 25 for i=2 terms alone.
    #
    # But wait - we might be able to use half-integer coefficients to
    # create new rank-1 terms that cover multiple i=2 entries.
    
    # Let's try a different approach: look for a 3x3 or 4x4 sub-problem
    # that can be decomposed with rank less than schoolbook.
    #
    # The i=2 slice is a 5x5 matrix multiplication (j x k -> output).
    # Rank of 5x5 matmul is known to be at most 49 (schoolbook) but
    # can be less. However, we need it to integrate with the full tensor.
    
    # Given time constraints, let's try a targeted approach:
    # Check if we can save rank by using a known 3x3 matmul with rank 23
    # for a subset of the i=2 terms.
    
    # For now, if no improvement found, return the rank-71 solution.
    # But let's also try: can we use a 2x5x5 Strassen-like block for i=2?
    
    # Actually, let's try to see if we can replace some schoolbook terms
    # with a more efficient decomposition of a sub-block.
    
    # Try: use a 2x2x2 Strassen block for (i=2, j=0,1, k=0,1)
    # This would cover 8 terms with rank 7, saving 1.
    # But i=2 is a single row, so we can't do 2x2 in i.
    
    # Alternative: look at the j-k plane for i=2.
    # We have 5x5 = 25 terms. Can we decompose 5x5 matmul with rank < 25?
    # Known: rank(5x5) <= 49 is trivial, but actual minimum is unknown.
    # However, we can try a block approach: split j into {0,1,2} and {3,4},
    # k into {0,1,2} and {3,4}.
    
    # For a 3x3 sub-block: use known rank-23 decomposition for 3x3 matmul.
    # But we need to verify this works.
    
    # Given the complexity and time budget, let's stick with the rank-71
    # solution which is verified correct.
    
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}

    # Try one more thing: check if we can find a rank-20 decomposition
    # for the 5x5 (i=2) slice that's compatible.
    # This is research-level, so we'll skip and return rank 71.
    
    return {"U": U, "V": V, "W": W}
