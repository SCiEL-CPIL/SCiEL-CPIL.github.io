"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6

    # We use a Strassen-based decomposition for 4x4 * 4x6.
    # A is 4x4, B is 4x6, C = A*B is 4x6.
    # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B into 2x2 blocks: B = [[B11, B12, B13], [B21, B22, B23]]
    # C = [[C11, C12, C13], [C21, C22, C23]]
    
    # For each column block t (0,1,2), we compute [C1t; C2t] = A * B_t
    # where B_t is 4x2 (top 2x2 and bottom 2x2).
    # [C1t; C2t] = [A11*B1t + A12*B2t; A21*B1t + A22*B2t]
    
    # For each (row_block, col_block) pair, we have a 2x4 * 4x2 product
    # which decomposes into two 2x2*2x2 products.
    # Using Strassen for each 2x2*2x2 gives 7 mults.
    # 6 such products * 7 = 42 mults total.
    
    # But we can share A-side combinations across column blocks.
    # Let's use a cleaner approach: for each of the 3 column blocks,
    # we need 2x2*2x2 products for (A11*B1t, A12*B2t, A21*B1t, A22*B2t)
    # That's 4 Strassen decompositions per column block = 28 per block * 3 = 84.
    # Wait, that's worse. Let me think again.
    
    # Actually, for a fixed column block t, we compute:
    # C1t = A11*B1t + A12*B2t  (2x2 = 2x2*2x2 + 2x2*2x2)
    # C2t = A21*B1t + A22*B2t  (2x2 = 2x2*2x2 + 2x2*2x2)
    
    # Each 2x2*2x2 can be done with 7 mults.
    # So per column block: 4 * 7 = 28 mults.
    # Total: 3 * 28 = 84 mults.
    
    # Alternatively, we can view it as:
    # For column block t, A (4x4) * B_t (4x2)
    # This is a 4x4 * 4x2 product.
    # We can use a Strassen-like approach for this.
    
    # Let me just implement the straightforward 42-rank approach:
    # For each of the 6 (row_block, col_block) pairs, use 7 Strassen mults.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 16 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen for 2x2 * 2x2 -> 2x2
    # Inputs: X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    # Z00 = M1 - M4 + M6 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M3 + M2 - M5
    
    # For each (rb, cb) where rb in {0,1}, cb in {0,1,2}:
    # A_rb = A[2*rb:2*rb+2, :]  (2x4)
    # B_cb = B[:, 2*cb:2*cb+2]  (4x2)
    # A_rb = [A_rb1, A_rb2] where A_rb1 = A[2*rb:2*rb+2, 0:2], A_rb2 = A[2*rb:2*rb+2, 2:4]
    # B_cb = [B_cb1; B_cb2] where B_cb1 = B[0:2, 2*cb:2*cb+2], B_cb2 = B[2:4, 2*cb:2*cb+2]
    # C_rb_cb = A_rb1 * B_cb1 + A_rb2 * B_cb2
    
    # For A_rb1 * B_cb1: use Strassen with X=A_rb1, Y=B_cb1
    # For A_rb2 * B_cb2: use Strassen with X=A_rb2, Y=B_cb2
    
    # Total 12 Strassen decompositions * 7 = 84? No, that's 12*7=84.
    # Wait, I said 42 earlier. Let me recount.
    # 2 row blocks * 3 col blocks = 6 pairs.
    # Each pair needs 2 Strassen decompositions (for the two 2x2*2x2 products).
    # 6 * 2 * 7 = 84.
    
    # Hmm, that's 84. Let me just use rank 84 for now.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for rb in range(2):
        for cb in range(3):
            # A_rb1 = A[2*rb:2*rb+2, 0:2]
            # A_rb2 = A[2*rb:2*rb+2, 2:4]
            # B_cb1 = B[0:2, 2*cb:2*cb+2]
            # B_cb2 = B[2:4, 2*cb:2*cb+2]
            
            # Strassen for A_rb1 * B_cb1
            # X = A_rb1, Y = B_cb1
            # x00 = A[2*rb, 0], x01 = A[2*rb, 1]
            # x10 = A[2*rb+1, 0], x11 = A[2*rb+1, 1]
            # y00 = B[0, 2*cb], y01 = B[0, 2*cb+1]
            # y10 = B[1, 2*cb], y11 = B[1, 2*cb+1]
            
            # M1 = (x00+x11)(y00+y11)
            U[2*rb * m + 0][col] += 1  # A[2*rb, 0]
            U[(2*rb+1) * m + 1][col] += 1  # A[2*rb+1, 1]
            V[0 * p + 2*cb][col] += 1  # B[0, 2*cb]
            V[1 * p + (2*cb+1)][col] += 1  # B[1, 2*cb+1]
            # C contribution: Z00 of this product contributes to C[2*rb, 2*cb]
            W[(2*cb) * n + 2*rb][col] += 1
            col += 1
            
            # M2 = (x10+x11)y00
            U[(2*rb+1) * m + 0][col] += 1  # A[2*rb+1, 0]
            U[(2*rb+1) * m + 1][col] += 1  # A[2*rb+1, 1]
            V[0 * p + 2*cb][col] += 1  # B[0, 2*cb]
            # Z10 contributes to C[2*rb+1, 2*cb]
            W[(2*cb) * n + (2*rb+1)][col] += 1
            col += 1
            
            # M3 = x00(y01-y11)
            U[2*rb * m + 0][col] += 1  # A[2*rb, 0]
            V[0 * p + (2*cb+1)][col] += 1  # B[0, 2*cb+1]
            V[1 * p + (2*cb+1)][col] -= 1  # B[1, 2*cb+1]
            # Z01 contributes to C[2*rb, 2*cb+1]
            W[(2*cb+1) * n + 2*rb][col] += 1
            col += 1
            
            # M4 = x11(y10-y00)
            U[(2*rb+1) * m + 1][col] += 1  # A[2*rb+1, 1]
            V[1 * p + 2*cb][col] += 1  # B[1, 2*cb]
            V[0 * p + 2*cb][col] -= 1  # B[0, 2*cb]
            # Z00 and Z10
            W[(2*cb) * n + 2*rb][col] -= 1
            W[(2*cb) * n + (2*rb+1)][col] += 1
            col += 1
            
            # M5 = (x00+x01)y11
            U[2*rb * m + 0][col] += 1  # A[2*rb, 0]
            U[2*rb * m + 1][col] += 1  # A[2*rb, 1]
            V[1 * p + (2*cb+1)][col] += 1  # B[1, 2*cb+1]
            # Z01 and Z11
            W[(2*cb+1) * n + 2*rb][col] += 1
            W[(2*cb+1) * n + (2*rb+1)][col] -= 1
            col += 1
            
            # M6 = (x10-x00)(y00+y01)
            U[(2*rb+1) * m + 0][col] += 1  # A[2*rb+1, 0]
            U[2*rb * m + 0][col] -= 1  # A[2*rb, 0]
            V[0 * p + 2*cb][col] += 1  # B[0, 2*cb]
            V[0 * p + (2*cb+1)][col] += 1  # B[0, 2*cb+1]
            # Z00
            W[(2*cb) * n + 2*rb][col] += 1
            col += 1
            
            # M7 = (x01-x11)(y10+y11)
            U[2*rb * m + 1][col] += 1  # A[2*rb, 1]
            U[(2*rb+1) * m + 1][col] -= 1  # A[2*rb+1, 1]
            V[1 * p + 2*cb][col] += 1  # B[1, 2*cb]
            V[1 * p + (2*cb+1)][col] += 1  # B[1, 2*cb+1]
            # Z00 and Z11
            W[(2*cb) * n + 2*rb][col] += 1
            W[(2*cb+1) * n + (2*rb+1)][col] -= 1
            col += 1
            
            # Now Strassen for A_rb2 * B_cb2
            # X = A_rb2 = A[2*rb:2*rb+2, 2:4]
            # Y = B_cb2 = B[2:4, 2*cb:2*cb+2]
            # x00 = A[2*rb, 2], x01 = A[2*rb, 3]
            # x10 = A[2*rb+1, 2], x11 = A[2*rb+1, 3]
            # y00 = B[2, 2*cb], y01 = B[2, 2*cb+1]
            # y10 = B[3, 2*cb], y11 = B[3, 2*cb+1]
            
            # M1 = (x00+x11)(y00+y11)
            U[2*rb * m + 2][col] += 1
            U[(2*rb+1) * m + 3][col] += 1
            V[2 * p + 2*cb][col] += 1
            V[3 * p + (2*cb+1)][col] += 1
            W[(2*cb) * n + 2*rb][col] += 1
            col += 1
            
            # M2 = (x10+x11)y00
            U[(2*rb+1) * m + 2][col] += 1
            U[(2*rb+1) * m + 3][col] += 1
            V[2 * p + 2*cb][col] += 1
            W[(2*cb) * n + (2*rb+1)][col] += 1
            col += 1
            
            # M3 = x00(y01-y11)
            U[2*rb * m + 2][col] += 1
            V[2 * p + (2*cb+1)][col] += 1
            V[3 * p + (2*cb+1)][col] -= 1
            W[(2*cb+1) * n + 2*rb][col] += 1
            col += 1
            
            # M4 = x11(y10-y00)
            U[(2*rb+1) * m + 3][col] += 1
            V[3 * p + 2*cb][col] += 1
            V[2 * p + 2*cb][col] -= 1
            W[(2*cb) * n + 2*rb][col] -= 1
            W[(2*cb) * n + (2*rb+1)][col] += 1
            col += 1
            
            # M5 = (x00+x01)y11
            U[2*rb * m + 2][col] += 1
            U[2*rb * m + 3][col] += 1
            V[3 * p + (2*cb+1)][col] += 1
            W[(2*cb+1) * n + 2*rb][col] += 1
            W[(2*cb+1) * n + (2*rb+1)][col] -= 1
            col += 1
            
            # M6 = (x10-x00)(y00+y01)
            U[(2*rb+1) * m + 2][col] += 1
            U[2*rb * m + 2][col] -= 1
            V[2 * p + 2*cb][col] += 1
            V[2 * p + (2*cb+1)][col] += 1
            W[(2*cb) * n + 2*rb][col] += 1
            col += 1
            
            # M7 = (x01-x11)(y10+y11)
            U[2*rb * m + 3][col] += 1
            U[(2*rb+1) * m + 3][col] -= 1
            V[3 * p + 2*cb][col] += 1
            V[3 * p + (2*cb+1)][col] += 1
            W[(2*cb) * n + 2*rb][col] += 1
            W[(2*cb+1) * n + (2*rb+1)][col] -= 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
