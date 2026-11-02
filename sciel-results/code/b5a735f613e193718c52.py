import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For 2x2 A, 2x2 B -> 2x2 C:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Strassen term: list of (a_idx, a_coeff), (b_idx, b_coeff), (c_idx, c_coeff)
    # A indices: (row, col) -> flat index row*m + col for m=2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: (row, col) -> flat index row*p_local + col for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: (row, col) -> flat index row*n + col for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00+, C11+
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2: (A10+A11)B00 -> C10+, C11-
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        # M3: A00(B01-B11) -> C01+, C11+
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        # M4: A11(B10-B00) -> C00+, C10+
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        # M5: (A00+A01)B11 -> C00-, C01+
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        # M6: (A10-A00)(B00+B01) -> C11+
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M7: (A01-A11)(B10+B11) -> C00+
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]
    
    U_rows = n * m  # 4
    V_rows = m * p  # 26
    W_rows = p * n  # 26
    
    # Collect all terms
    all_U = []  # list of (row, coeff)
    all_V = []
    all_W = []
    
    # Process 6 pairs of columns: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for pair_idx in range(6):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        
        for term in strassen_terms:
            a_parts, b_parts, c_parts = term
            u_entries = []
            v_entries = []
            w_entries = []
            
            for (ai, aj), coeff in a_parts:
                u_entries.append((ai * m + aj, coeff))
            
            for (bi, bj), coeff in b_parts:
                # B index: bi is row (0 or 1), bj is column within the 2x2 block
                # Actual column in B: k1 + bj
                v_entries.append((bi * p + (k1 + bj), coeff))
            
            for (ci, cj), coeff in c_parts:
                # C index: ci is row, cj is column within the 2x2 block
                # Actual column in C: k1 + cj
                # W index: (k1 + cj) * n + ci
                w_entries.append(((k1 + cj) * n + ci, coeff))
            
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)
    
    # Process single column k=12
    for i in range(n):
        for j in range(m):
            all_U.append([(i * m + j, 1)])
            all_V.append([(j * p + 12, 1)])
            all_W.append([(12 * n + i, 1)])
    
    rank = len(all_U)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for (row, coeff) in all_U[c]:
            U[row][c] = coeff
        for (row, coeff) in all_V[c]:
            V[row][c] = coeff
        for (row, coeff) in all_W[c]:
            W[row][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
