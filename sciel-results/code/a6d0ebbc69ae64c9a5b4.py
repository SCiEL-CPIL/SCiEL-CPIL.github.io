import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # Use the 4x4 matrix multiplication algorithm (Lambert, rank 19) to compute
    # the product of a 2x2 matrix A and a 2x14 matrix B.
    # Embed B into a 4x4 matrix by placing it in the top-left 2x4 block:
    #   B_4 = [[B[0,0:4], B[0,4:8]],
    #          [B[1,0:4], B[1,4:8]],
    #          [0, 0, 0, 0],
    #          [0, 0, 0, 0]]
    # But that's not a valid 4x4 embedding for a single multiplication.
    # Instead, split B into two 2x4 blocks: B = [B1 | B2]
    # We can compute A*B1 and A*B2 separately, but that doubles the rank.
    # 
    # Better approach: embed A and B into 4x4 matrices such that
    # the top-left 2x4 block of the product is A*B.
    # Let A_4 = [[A, 0], [0, 0]] (4x4)
    # Let B_4 = [[B, 0], [0, 0]] but B is 2x4, so we need to pad to 2x4 -> 4x4.
    # Actually, we have 7 blocks of 2x2. Let's use 4x4 multiplication for each pair of blocks?
    # No, that's what Strassen does.
    #
    # Alternative: Use the fact that 2x2x14 can be seen as 2x2x(2*7).
    # The current approach uses 7 Strassen terms per 2x2 block, giving rank 49.
    #
    # Let's try a different embedding: treat the 2x14 matrix as a 2x2 block matrix
    # with 7 blocks. We can use a 4x4 algorithm that handles multiple blocks at once.
    #
    # Actually, let's just verify the current Strassen decomposition is correct
    # and see if we can find a lower rank.
    #
    # For 2x2x14, the trivial rank is 2*2*14 = 56.
    # Strassen gives 7 * 7 = 49.
    #
    # Is there a rank-42 algorithm? That would be 6 * 7 = 42, but we need 7 terms per block.
    # Or maybe a mixed algorithm.
    #
    # Let's stick with the Strassen approach but verify it's correct.
    
    num_blocks = p // 2  # 7 blocks
    rank = 7 * num_blocks  # 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication
    # Local indices: U: 0=A00, 1=A01, 2=A10, 3=A11
    # V: 0=B00, 1=B01, 2=B10, 3=B11
    # W: 0=C00, 1=C01, 2=C10, 3=C11
    
    terms = [
        # M1: (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    col = 0
    for b in range(num_blocks):
        # Global V indices for block b
        # B00: row 0, col 2b -> 14*0 + 2b
        # B01: row 0, col 2b+1 -> 14*0 + 2b+1
        # B10: row 1, col 2b -> 14*1 + 2b
        # B11: row 1, col 2b+1 -> 14*1 + 2b+1
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        
        # Global W indices for block b
        # C00: row 2b, col 0 -> 2*(2b) + 0
        # C01: row 2b+1, col 0 -> 2*(2b+1) + 0
        # C10: row 2b, col 1 -> 2*(2b) + 1
        # C11: row 2b+1, col 1 -> 2*(2b+1) + 1
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for l_idx, val in v_dict.items():
                V[v_offsets[l_idx]][col] += val
            for l_idx, val in w_dict.items():
                W[w_offsets[l_idx]][col] += val
            col += 1
    
    return {"U": U, "V": V, "W": W}
