"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 4 independent Strassen 2x2x2 instances.
    # Each Strassen instance has rank 7, so total rank = 4 * 7 = 28.
    # 
    # The 4x4 matrix A is partitioned into four 2x2 blocks:
    #   A = [[A00, A01],
    #        [A10, A11]]
    # The 4x6 matrix B is partitioned into four 2x3 blocks:
    #   B = [[B00, B01],
    #        [B10, B11]]
    # where B00, B10 are 2x3 and B01, B11 are 2x3.
    # 
    # The product C = A*B has shape 4x6, partitioned into four 2x3 blocks:
    #   C = [[C00, C01],
    #        [C10, C11]]
    # 
    # We can compute each C block using Strassen on 2x2x2, but that doesn't
    # directly work because B has 3 columns. Instead, we handle the 6 columns
    # by splitting B into three 4x2 matrices (each with 2 columns), or we
    # handle it differently.
    #
    # Actually, a cleaner approach: treat the 4x4x6 multiplication as a
    # tensor product. We can decompose it as 4 independent 2x2x2 Strassen
    # instances if we view the problem correctly.
    #
    # Let me think more carefully. The standard schoolbook rank is 4*4*6=96.
    # A rank-7 Strassen for 2x2x2 gives rank 7 for multiplying 2x2 by 2x2.
    # For 4x4x6, we can tile: 4x4 = (2x2) x (2x2) blocks, and 6 columns
    # can be handled as 3 groups of 2 columns.
    #
    # So we have a 2x2 grid of block-multiplications, each of which is a
    # 2x2 * 2x2 -> 2x2 problem (since each B block is 2x2 when we split
    # the 6 columns into 3 pairs). Total: 4 block positions * 3 column
    # pairs = 12 Strassen instances? No, that's not right either.
    #
    # Let me reconsider. The output C is 4x6. We can write C[i,k] for
    # i in 0..3, k in 0..5. The tensor T[i,j,k] = 1 for all i,j,k.
    #
    # A valid decomposition uses rank r tensors (U_c, V_c, W_c) such that
    # T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i].
    #
    # For Strassen 2x2x2, the 7 rank-1 terms are well-known. Let me define
    # them explicitly for a 2x2x2 block, then tile.
    #
    # Strassen's 7 products for 2x2x2 (A 2x2, B 2x2, C=A*B 2x2):
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B00)
    # m5 = (A00+A10)B11
    # m6 = (A11+A01)(B00+B10)
    # m7 = (A01-A11)(B11+B10)
    #
    # C00 = m1 - m4 + m5 + m7
    # C01 = m3 + m4
    # C10 = m2 + m5
    # C11 = m1 - m3 + m2 - m6
    #
    # Now for the 4x4x6 case, I'll use 4 independent Strassen instances
    # corresponding to the 4 quadrant pairs of (row-block, col-block).
    #
    # Actually, the simplest correct approach with a clear rank reduction:
    # Use 3 copies of the 4x4x2 Strassen-based decomposition.
    #
    # For 4x4x2: We can use 4 Strassen 2x2x2 instances (one per 2x2 block
    # of A times a 2x2 block of B). Each has rank 7, total rank 28 per
    # 4x4x2. For 4x4x6 = 3 * 4x4x2, total rank = 84.
    #
    # But wait, we can do better. Let me just implement 4 Strassen instances
    # for the 4x4x6 by treating it as: the 6 columns don't need to be paired.
    #
    # The most straightforward improvement: use the known rank-7 2x2x2
    # Strassen and apply it to each of the 4 (2x2 A-block) * (2x2 B-subblock)
    # combinations. Since B is 4x6, we split it into 3 horizontal strips of
    # 4x2. For each strip, we have 4 block-multiplications (each 2x2 * 2x2),
    # each using Strassen rank 7. Total per strip: 28. Total: 84.
    #
    # This gives rank 84, which is much better than 96.

    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 basis vectors.
    # For a 2x2x2 block with row-offset ir (0 or 1 in the 2x2 A), 
    # col-offset jc (0 or 1 in the 2x2 B), and output offset (oc, oc_row):
    # U index: (ir*2 + 0, ir*2 + 1) within the 2x2 A block
    # V index: (jc*2 + 0, jc*2 + 1) within the 2x2 B block  
    # W index: (oc*2 + 0, oc*2 + 1) within the 2x2 C block
    
    # The 7 Strassen products, defined as (U_coeffs, V_coeffs, W_coeffs)
    # where each is a 4-vector for the 2x2 case.
    # U: indexed by [a00, a01, a10, a11] -> [i*2+j for i,j in 0,1]
    # V: indexed by [b00, b01, b10, b11] -> [j*2+k for j,k in 0,1]
    # W: indexed by [c00, c01, c10, c11] -> [k*2+i for k,i in 0,1]
    
    strassen_terms = [
        # m1 = (a00+a11)(b00+b11) -> c00 += m1, c11 += m1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # m2 = (a10+a11)b00 -> c10 += m2, c11 -= m2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # m3 = a00(b10-b11) -> c01 += m3, c11 += m3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # m4 = a11(b01-b00) -> c00 -= m4, c01 += m4
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 1, 0, 0]),
        # m5 = (a00+a10)b11 -> c00 += m5, c10 += m5
        ([1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0]),
        # m6 = (a11+a01)(b00+b10) -> c11 -= m6
        ([0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, -1]),
        # m7 = (a01-a11)(b11+b10) -> c00 += m7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for strip in range(3):  # 3 strips of 2 columns each
        for block_row in range(2):  # 2 row-blocks of A (2x2 each)
            for block_col in range(2):  # 2 col-blocks of B (2x2 each)
                # Row offset in A: block_row * 2
                # Col offset in B: block_col * 2 + strip * 2
                # Output row offset: block_row * 2
                # Output col offset: strip * 2
                
                a_row0 = block_row * 2
                b_col0 = block_col * 2 + strip * 2
                c_row0 = block_row * 2
                c_col0 = strip * 2
                
                for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                    for idx in range(4):
                        # U: A block element (a_row0 + i, a_col + j) where i,j in 0,1
                        u_local_idx = idx  # 0=a00, 1=a01, 2=a10, 3=a11
                        i_local = u_local_idx // 2
                        j_local = u_local_idx % 2
                        u_global_idx = (a_row0 + i_local) * m + (block_col * 2 + j_local)
                        U[u_global_idx][col] = u_coeffs[idx]
                        
                        # V: B block element (b_row + j, b_col + k) where j,k in 0,1
                        v_local_idx = idx  # 0=b00, 1=b01, 2=b10, 3=b11
                        j_local_v = v_local_idx // 2
                        k_local_v = v_local_idx % 2
                        v_global_idx = (block_row * 2 + j_local_v) * p + (b_col0 + k_local_v)
                        V[v_global_idx][col] = v_coeffs[idx]
                        
                        # W: C block element (c_row + i, c_col + k) where i,k in 0,1
                        # W is indexed by [k*n + i]
                        w_local_idx = idx  # 0=c00, 1=c01, 2=c10, 3=c11
                        k_local_w = w_local_idx // 2
                        i_local_w = w_local_idx % 2
                        w_global_idx = (c_col0 + k_local_w) * n + (c_row0 + i_local_w)
                        W[w_global_idx][col] = w_coeffs[idx]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
