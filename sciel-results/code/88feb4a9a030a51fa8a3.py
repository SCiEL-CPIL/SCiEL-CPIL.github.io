"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen-like blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for M_{2,2,2}
    # u: flattened A (2x2), v: flattened B (2x2), w: flattened C (2x2)
    # Indices: 00, 01, 10, 11
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00 +, C11 -
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
        # M2: (A10+A11)B00 -> C10 +, C00 -
        ([0, 0, 1, 1], [1, 0, 0, 0], [-1, 0, 1, 0]),
        # M3: A00(B01-B11) -> C01 +, C11 -
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]),
        # M4: A11(B10-B00) -> C10 +, C01 ? No.
        # C10 = M2 + M4. C00 = M1 - M2 + M4 - M5.
        # M4 contributes to C10 (+) and C00 (+)?
        # Let's re-verify Strassen formulas.
        # C00 = M1 - M2 + M4 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M7
        # M4: A11 * (B10 - B00).
        # u: [0, 0, 0, 1], v: [0, 0, 1, -1], w: [1, 0, 1, 0] (C00 +, C10 +)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5: (A00+A01)B11 -> C01 +, C00 -
        # u: [1, 1, 0, 0], v: [0, 0, 0, 1], w: [-1, 1, 0, 0] (C00 -, C01 +)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (A10-A00)(B00+B01) -> C11 -
        # u: [0, 0, 1, -1]? No, A10 - A00 is [ -1, 0, 1, 0 ]?
        # A indices: 00, 01, 10, 11.
        # A10 is idx 2. A00 is idx 0.
        # u: [-1, 0, 1, 0]. v: [1, 1, 0, 0]. w: [0, 0, 0, -1] (C11 -)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, -1]),
        # M7: (A01-A11)(B10+B11) -> C11 -
        # A01 idx 1, A11 idx 3. u: [0, 1, 0, -1]. v: [0, 0, 1, 1]. w: [0, 0, 0, -1] (C11 -)
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, -1]),
    ]
    
    col = 0
    # 2 parts (A1, A2)
    for part in range(2):
        # 4 blocks (B cols)
        for block in range(4):
            for term_idx in range(7):
                u, v, w = strassen_terms[term_idx]
                
                # Map u (2x2) to U (2x4)
                # part 0: cols 0,1. part 1: cols 2,3.
                # u indices: 00, 01, 10, 11 -> global rows 0,1.
                # u[0] -> A[0, part*2]
                # u[1] -> A[0, part*2+1]
                # u[2] -> A[1, part*2]
                # u[3] -> A[1, part*2+1]
                U[0 * m + part*2 + 0][col] = u[0]
                U[0 * m + part*2 + 1][col] = u[1]
                U[1 * m + part*2 + 0][col] = u[2]
                U[1 * m + part*2 + 1][col] = u[3]
                
                # Map v (2x2) to V (4x8)
                # part 0: rows 0,1. part 1: rows 2,3.
                # block: cols block*2, block*2+1.
                # v[0] -> B[part*2 + 0, block*2 + 0]
                # v[1] -> B[part*2 + 0, block*2 + 1]
                # v[2] -> B[part*2 + 1, block*2 + 0]
                # v[3] -> B[part*2 + 1, block*2 + 1]
                V[(part*2 + 0) * p + block*2 + 0][col] = v[0]
                V[(part*2 + 0) * p + block*2 + 1][col] = v[1]
                V[(part*2 + 1) * p + block*2 + 0][col] = v[2]
                V[(part*2 + 1) * p + block*2 + 1][col] = v[3]
                
                # Map w (2x2) to W (8x2)
                # Output C is 2x8.
                # block: cols block*2, block*2+1.
                # w[0] -> C[0, block*2] -> W[block*2 * n + 0]
                # w[1] -> C[0, block*2+1] -> W[(block*2+1) * n + 0]
                # w[2] -> C[1, block*2] -> W[block*2 * n + 1]
                # w[3] -> C[1, block*2+1] -> W[(block*2+1) * n + 1]
                W[block*2 * n + 0][col] = w[0]
                W[(block*2+1) * n + 0][col] = w[1]
                W[block*2 * n + 1][col] = w[2]
                W[(block*2+1) * n + 1][col] = w[3]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
