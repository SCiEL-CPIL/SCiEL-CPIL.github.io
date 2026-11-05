"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 tensor decomposition for 6x6 matrix multiplication.
    
    Approach: 1-level Strassen on 6x6 matrices split into 2x2 blocks of 3x3.
    Each of the 7 Strassen terms is a 3x3 matrix multiplication, decomposed
    via schoolbook (rank 27). Total rank = 7 * 27 = 189.
    
    The tensor decomposition satisfies:
      C[i,k] = sum over r: U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
    for all i, j, k in 0..5, where C = A * B.
    """
    n = m = p = 6
    rank = 189  # 7 Strassen terms * 27 schoolbook terms per 3x3 product
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    # Split 6x6 into 2x2 blocks of 3x3:
    # Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
    # br, bc in {0, 1}
    
    # Strassen terms: each is (left_combo, right_combo)
    # left_combo: dict from (br, bc) to coeff, representing linear combo of A blocks
    # right_combo: dict from (br, bc) to coeff, representing linear combo of B blocks
    #
    # T1: (A00+A11) * (B00+B11)
    # T2: (A10+A11) * B00
    # T3: A00 * (B01-B11)
    # T4: A11 * (B10-B00)
    # T5: (A00+A01) * B11
    # T6: (A10-A00) * (B00+B01)
    # T7: (A01-A11) * (B10+B11)
    
    left_combos = [
        {(0,0): 1, (1,1): 1},   # T1
        {(1,0): 1, (1,1): 1},   # T2
        {(0,0): 1},              # T3
        {(1,1): 1},              # T4
        {(0,0): 1, (0,1): 1},   # T5
        {(1,0): 1, (0,0): -1},  # T6
        {(0,1): 1, (1,1): -1},  # T7
    ]
    right_combos = [
        {(0,0): 1, (1,1): 1},   # T1
        {(0,0): 1},              # T2
        {(0,1): 1, (1,1): -1},  # T3
        {(1,0): 1, (0,0): -1},  # T4
        {(1,1): 1},              # T5
        {(0,0): 1, (0,1): 1},   # T6
        {(1,0): 1, (1,1): 1},   # T7
    ]
    
    # Output contributions: which Strassen terms contribute to which output block
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    output_contribs = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0, 1): [(2, 1), (4, 1)],
        (1, 0): [(1, 1), (3, 1)],
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    # Precompute: for each Strassen term t, list of (rbr, rbc, sign)
    term_to_outputs = [[] for _ in range(7)]
    for (rbr, rbc), contribs in output_contribs.items():
        for (t, sign) in contribs:
            term_to_outputs[t].append((rbr, rbc, sign))
    
    col = 0
    for t in range(7):
        lc = left_combos[t]
        rc = right_combos[t]
        outputs = term_to_outputs[t]
        
        # Schoolbook decomposition of the 3x3 product X*Y:
        # Z[i,k] = sum_a X[i,a] * Y[a,k]
        # Each (i, a, k) triple gives one rank-1 term
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # U: for each block (br, bc) in left_combo with coeff cl,
                    # X[i,a] includes cl * A[3*br+i, 3*bc+a]
                    # U row index = (3*br+i) * 6 + (3*bc+a)
                    for (br, bc), cl in lc.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += cl
                    
                    # V: for each block (br, bc) in right_combo with coeff cr,
                    # Y[a,k] includes cr * B[3*br+a, 3*bc+k]
                    # V row index = (3*br+a) * 6 + (3*bc+k)
                    for (br, bc), cr in rc.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += cr
                    
                    # W: this term contributes to output C[i_tot, k_tot]
                    # where i_tot = 3*rbr + i, k_tot = 3*rbc + k
                    # W row index = k_tot * 6 + i_tot (trace convention W[k*n+i, r])
                    for (rbr, rbc, sign) in outputs:
                        i_tot = 3 * rbr + i
                        k_tot = 3 * rbc + k
                        W[k_tot * 6 + i_tot][col] += sign
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
