import numpy as np
import random
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    
    # We will use a decomposition based on splitting the 5x5 matrices into 4x4 blocks and vectors.
    # A = [[A', a], [c, e]] where A' is 4x4, a is 4x1, c is 1x4, e is scalar.
    # B = [[B', b], [d, f]] where B' is 4x4, b is 1x4, d is 4x1, f is scalar.
    # Note: In standard matrix multiplication C = AB:
    # C = [[A'B' + ab, A'd + af], [cB' + eb, cd + ef]]
    # Indices:
    # A' is A[0:4, 0:4]
    # a is A[0:4, 4]
    # c is A[4, 0:4]
    # e is A[4, 4]
    # B' is B[0:4, 0:4]
    # b is B[4, 0:4]  (Row 4 of B)
    # d is B[0:4, 4]  (Col 4 of B)
    # f is B[4, 4]
    
    # C[0:4, 0:4] = A'B' + ab
    # C[0:4, 4]   = A'd + af
    # C[4, 0:4]   = cB' + eb
    # C[4, 4]     = cd + ef
    
    # We use Strassen's algorithm for the 4x4 multiplication (A'B'), which decomposes into 7 multiplications of 2x2 blocks.
    # Each 2x2 multiplication is done via schoolbook (rank 4).
    # Total rank for A'B' is 7 * 4 = 28.
    
    # The other terms are simple bilinear forms:
    # ab: 4*1 * 1*4 -> 16 terms
    # A'd: 4*4 * 4*1 -> 16 terms
    # af: 4*1 * 1*1 -> 4 terms
    # cB': 1*4 * 4*4 -> 16 terms
    # eb: 1*1 * 1*4 -> 4 terms
    # cd: 1*4 * 4*1 -> 4 terms
    # ef: 1*1 * 1*1 -> 1 term
    
    total_rank = 28 + 16 + 16 + 4 + 16 + 4 + 4 + 1
    # total_rank = 89
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # 1. A'B' using Strassen on 2x2 blocks
    # A' blocks:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B' blocks:
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    # Strassen M1-M7 are 2x2 matrix products.
    # We will generate the 4 schoolbook terms for each M.
    
    def get_2x2_indices(row_offset, col_offset):
        # Returns list of (u_idx, v_idx, w_idx) for the 2x2 block at (row_offset, col_offset) in A, B, C
        # A block is at rows row_offset, cols col_offset
        # B block is at rows row_offset, cols col_offset
        # C block is at rows col_offset (result col), cols row_offset (result row)
        # Wait, C = AB. C[i,k] = sum_j A[i,j] B[j,k].
        # A block (i,j), B block (j,k) -> C block (i,k).
        # In our 5x5 grid:
        # A indices: i, j. Flat: i*5+j
        # B indices: j, k. Flat: j*5+k
        # C indices: k, i. Flat: k*5+i
        
        # The 2x2 block of A at (r, c) corresponds to i in r..r+1, j in c..c+1.
        # The 2x2 block of B at (c, k_offset) corresponds to j in c..c+1, k in k_offset..k_offset+1.
        # The 2x2 block of C at (r, k_offset) corresponds to i in r..r+1, k in k_offset..k_offset+1.
        
        terms = []
        for di in range(2):
            for dj in range(2):
                i = r + di
                j = c + dj
                for dk in range(2):
                    k = k_offset + dk
                    # A[i,j] * B[j,k] -> C[k,i]
                    u = i * 5 + j
                    v = j * 5 + k
                    w = k * 5 + i
                    terms.append((u, v, w))
        return terms

    # Strassen M1 = (A00 + A11)(B00 + B11)
    # We need to decompose (X+Y)(Z+W) = XZ + XW + YZ + YW
    # X=A00, Y=A11, Z=B00, W=B11
    
    # M1 terms:
    # A00*B00
    for (u,v,w) in get_2x2_indices(0, 0, 0): # A(0,0), B(0,0) -> C(0,0) block
        add_term(u, v, w)
    # A00*B11
    for (u,v,w) in get_2x2_indices(0, 0, 2): # A(0,0), B(0,2) -> C(0,2) block
        add_term(u, v, w)
    # A11*B00
    for (u,v,w) in get_2x2_indices(2, 2, 0): # A(2,2), B(2,0) -> C(2,0) block
        add_term(u, v, w)
    # A11*B11
    for (u,v,w) in get_2x2_indices(2, 2, 2): # A(2,2), B(2,2) -> C(2,2) block
        add_term(u, v, w)

    # M2 = (A10 + A11)B00 = A10B00 + A11B00
    # A10B00
    for (u,v,w) in get_2x2_indices(2, 0, 0): # A(2,0), B(2,0) -> C(2,0) block
        add_term(u, v, w)
    # A11B00
    for (u,v,w) in get_2x2_indices(2, 2, 0): # A(2,2), B(2,0) -> C(2,0) block
        add_term(u, v, w)

    # M3 = A00(B01 - B11) = A00B01 - A00B11
    # A00B01
    for (u,v,w) in get_2x2_indices(0, 0, 2): # A(0,0), B(0,2) -> C(0,2) block
        add_term(u, v, w)
    # -A00B11
    for (u,v,w) in get_2x2_indices(0, 0, 2): # A(0,0), B(0,2) -> C(0,2) block
        add_term(u, v, w, -1) # Wait, B11 is at (2,2). B01 is at (0,2).
        # Correction: B11 is B[2:4, 2:4]. So indices (2,2).
        # My get_2x2_indices(r, c, k_offset) assumes B is at (c, k_offset).
        # For A00B11: A is (0,0), B is (2,2).
        # So get_2x2_indices(0, 0, 2) is A(0,0) and B(0,2). This is B01.
        # I need B(2,2).
        
    # Let's restart the Strassen part with clearer logic.
    # I will define the 7 Strassen multiplications explicitly by summing the schoolbook terms for the constituent 2x2 blocks.
    
    # Reset col to 0 and rebuild
    col = 0
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    def add_block_product(a_r, a_c, b_r, b_c, sign=1):
        # Adds schoolbook decomposition for A[a_r:a_r+2, a_c:a_c+2] * B[b_r:b_r+2, b_c:b_c+2]
        # Result is added to C[a_r:a_r+2, b_c:b_c+2]
        for di in range(2):
            for dj in range(2):
                i = a_r + di
                j = a_c + dj
                for dk in range(2):
                    k = b_c + dk
                    # A[i,j] * B[j,k]
                    # Note: B index is j (row of B) and k (col of B).
                    # In B block, row is b_r + (j - a_c)? No.
                    # The block B[b_r:b_r+2, b_c:b_c+2] has local indices.
                    # If we multiply A block at (a_r, a_c) and B block at (b_r, b_c),
                    # we require a_c == b_r for the dimensions to match in the product?
                    # Yes, A is n x m, B is m x p.
                    # A block rows a_r, cols a_c.
                    # B block rows b_r, cols b_c.
                    # For A*B to work, the col range of A must match row range of B.
                    # So a_c must equal b_r.
                    
                    u = i * 5 + (a_c + dj) # A[i, j] where j is global col index
                    v = (b_r + dj) * 5 + k # B[j, k] where j is global row index
                    w = k * 5 + i # C[k, i]
                    add_term(u, v, w, sign)

    # Strassen M1 = (A00 + A11)(B00 + B11) = A00B00 + A00B11 + A11B00 + A11B11
    # A00: r=0, c=0. A11: r=2, c=2.
    # B00: r=0, c=0. B11: r=2, c=2.
    add_block_product(0, 0, 0, 0, 1) # A00 B00
    add_block_product(0, 0, 0, 2, 1) # A00 B11? No, B11 is at r=2, c=2.
    # Wait, B11 is B[2:4, 2:4]. So b_r=2, b_c=2.
    # A00 is A[0:2, 0:2]. a_r=0, a_c=0.
    # Product A00 * B11 requires a_c (0) == b_r (2)? No.
    # Strassen works on 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]].
    # B = [[B00, B01], [B10, B11]].
    # A00 is top-left. A11 is bottom-right.
    # B00 is top-left. B11 is bottom-right.
    # M1 = (A00+A11)(B00+B11).
    # This is a product of two 2x2 matrices.
    # Let X = A00+A11, Y = B00+B11.
    # X is 2x2, Y is 2x2.
    # We need to decompose X*Y.
    # X[i,j] = A00[i,j] + A11[i,j].
    # Y[i,j] = B00[i,j] + B11[i,j].
    # (X*Y)[i,k] = sum_j X[i,j] Y[j,k].
    # This involves 4*4=16 scalar multiplications if done directly?
    # No, we can use schoolbook for the 2x2 product X*Y.
    # Schoolbook for 2x2:
    # C00 = X00Y00 + X01Y10
    # C01 = X00Y01 + X01Y11
    # C10 = X10Y00 + X11Y10
    # C11 = X10Y01 + X11Y11
    # Each term like X00Y00 is (A0000+A1100)(B0000+B1100).
    # This expands to 4 terms.
    # So M1 has 4*4 = 16 terms?
    # Yes, if we expand fully.
    # But we can just add the terms.
    
    # Let's implement a generic add_2x2_product(X_blocks, Y_blocks, C_offset)
    # Where X_blocks is a list of (r, c, sign) representing X = sum sign * A[r:r+2, c:c+2]
    # And Y_blocks is a list of (r, c, sign) representing Y = sum sign * B[r:r+2, c:c+2]
    # And C_offset is (r, c) for the resulting C block.
    
    def add_2x2_product(X_blocks, Y_blocks, c_r, c_c):
        # X and Y are 2x2 matrices.
        # We perform schoolbook multiplication of X and Y.
        # X[i,j] = sum_{bx in X_blocks} bx_sign * A[bx_r + i, bx_c + j]
        # Y[j,k] = sum_{by in Y_blocks} by_sign * B[by_r + j, by_c + k]
        # C[i,k] += sum_j X[i,j] Y[j,k]
        
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    # Term X[i,j] * Y[j,k]
                    for bx_r, bx_c, bx_sign in X_blocks:
                        for by_r, by_c, by_sign in Y_blocks:
                            # A index: (bx_r + i, bx_c + j)
                            # B index: (by_r + j, by_c + k)
                            # C index: (c_r + i, c_c + k)
                            a_i = bx_r + i
                            a_j = bx_c + j
                            b_j = by_r + j
                            b_k = by_c + k
                            c_i = c_r + i
                            c_k = c_c + k
                            
                            u = a_i * 5 + a_j
                            v = b_j * 5 + b_k
                            w = c_k * 5 + c_i
                            add_term(u, v, w, bx_sign * by_sign)

    # M1: X = A00 + A11, Y = B00 + B11. Result C00.
    # A00: (0,0,1), A11: (2,2,1)
    # B00: (0,0,1), B11: (2,2,1)
    add_2x2_product([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], 0, 0)
    
    # M2: X = A10 + A11, Y = B00. Result C10.
    # A10: (2,0,1), A11: (2,2,1)
    # B00: (0,0,1)
    add_2x2_product([(2,0,1), (2,2,1)], [(0,0,1)], 2, 0)
    
    # M3: X = A00, Y = B01 - B11. Result C01.
    # A00: (0,0,1)
    # B01: (0,2,1), B11: (2,2,-1)
    add_2x2_product([(0,0,1)], [(0,2,1), (2,2,-1)], 0, 2)
    
    # M4: X = A11, Y = B10 - B00. Result C11.
    # A11: (2,2,1)
    # B10: (2,0,1), B00: (0,0,-1)
    add_2x2_product([(2,2,1)], [(2,0,1), (0,0,-1)], 2, 2)
    
    # M5: X = A00 + A01, Y = B11. Result C11.
    # A00: (0,0,1), A01: (0,2,1)
    # B11: (2,2,1)
    add_2x2_product([(0,0,1), (0,2,1)], [(2,2,1)], 2, 2)
    
    # M6: X = A10 - A00, Y = B00 + B01. Result C11.
    # A10: (2,0,1), A00: (0,0,-1)
    # B00: (0,0,1), B01: (0,2,1)
    add_2x2_product([(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], 2, 2)
    
    # M7: X = A01 - A11, Y = B10 + B11. Result C01.
    # A01: (0,2,1), A11: (2,2,-1)
    # B10: (2,0,1), B11: (2,2,1)
    add_2x2_product([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], 0, 2)

    # Now we need to sum the contributions to C blocks.
    # C00 = M1
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Wait, the standard Strassen formulas are:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My add_2x2_product added directly to the C block.
    # This is incorrect because I added M1 to C00, but M1 also contributes to C11 with a negative sign.
    # I should have accumulated the M's separately or adjusted signs.
    # Let's restart with a cleaner approach:
    # Calculate the 7 M's as intermediate tensors? No, just add to U,V,W with correct signs.
    
    col = 0
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    def add_2x2_product(X_blocks, Y_blocks, c_r, c_c, sign=1):
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    for bx_r, bx_c, bx_sign in X_blocks:
                        for by_r, by_c, by_sign in Y_blocks:
                            a_i = bx_r + i
                            a_j = bx_c + j
                            b_j = by_r + j
                            b_k = by_c + k
                            c_i = c_r + i
                            c_k = c_c + k
                            
                            u = a_i * 5 + a_j
                            v = b_j * 5 + b_k
                            w = c_k * 5 + c_i
                            add_term(u, v, w, sign * bx_sign * by_sign)

    # M1
    add_2x2_product([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], 0, 0, 1) # +M1 to C00
    add_2x2_product([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], 2, 2, 1) # +M1 to C11
    
    # M2
    add_2x2_product([(2,0,1), (2,2,1)], [(0,0,1)], 2, 0, 1) # +M2 to C10
    add_2x2_product([(2,0,1), (2,2,1)], [(0,0,1)], 2, 2, -1) # -M2 to C11
    
    # M3
    add_2x2_product([(0,0,1)], [(0,2,1), (2,2,-1)], 0, 2, 1) # +M3 to C01
    add_2x2_product([(0,0,1)], [(0,2,1), (2,2,-1)], 2, 2, 1) # +M3 to C11
    
    # M4
    add_2x2_product([(2,2,1)], [(2,0,1), (0,0,-1)], 2, 0, 1) # +M4 to C10
    add_2x2_product([(2,2,1)], [(2,0,1), (0,0,-1)], 0, 0, 1) # +M4 to C00
    
    # M5
    add_2x2_product([(0,0,1), (0,2,1)], [(2,2,1)], 2, 2, 1) # +M5 to C11
    add_2x2_product([(0,0,1), (0,2,1)], [(2,2,1)], 0, 0, -1) # -M5 to C00
    
    # M6
    add_2x2_product([(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], 2, 2, 1) # +M6 to C11
    
    # M7
    add_2x2_product([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], 0, 2, 1) # +M7 to C01
    add_2x2_product([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], 0, 0, 1) # +M7 to C00

    # Now add the other terms
    # ab: a=A[0:4,4], b=B[4,0:4]
    for i in range(4):
        for k in range(4):
            u = i * 5 + 4
            v = 4 * 5 + k
            w = k * 5 + i
            add_term(u, v, w, 1)
            
    # A'd: A[0:4,0:4] * d=B[0:4,4]
    for i in range(4):
        for j in range(4):
            for k in range(1): # k is 4
                u = i * 5 + j
                v = j * 5 + 4
                w = 4 * 5 + i
                add_term(u, v, w, 1)
                
    # af: a=A[0:4,4], f=B[4,4]
    for i in range(4):
        u = i * 5 + 4
        v = 4 * 5 + 4
        w = 4 * 5 + i
        add_term(u, v, w, 1)
        
    # cB': c=A[4,0:4], B[0:4,0:4]
    for j in range(4):
        for k in range(4):
            u = 4 * 5 + j
            v = j * 5 + k
            w = k * 5 + 4
            add_term(u, v, w, 1)
            
    # eb: e=A[4,4], b=B[4,0:4]
    for k in range(4):
        u = 4 * 5 + 4
        v = 4 * 5 + k
        w = k * 5 + 4
        add_term(u, v, w, 1)
        
    # cd: c=A[4,0:4], d=B[0:4,4]
    for j in range(4):
        u = 4 * 5 + j
        v = j * 5 + 4
        w = 4 * 5 + 4
        add_term(u, v, w, 1)
        
    # ef: e=A[4,4], f=B[4,4]
    u = 4 * 5 + 4
    v = 4 * 5 + 4
    w = 4 * 5 + 4
    add_term(u, v, w, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
