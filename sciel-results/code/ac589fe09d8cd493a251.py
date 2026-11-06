import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    t0 = time.time()
    DEADLINE = 115.0

    n, m, p = DIMENSIONS
    # We use a 4x4 grid of 2x2 blocks, padded with zeros.
    # Block size is 2x2.
    # Grid size is 4x4.
    # Laderman's algorithm uses 21 multiplications for 4x4 matrices.
    # Each multiplication is a 2x2 matrix multiplication, done via Strassen (7 scalar mults).
    # Total rank = 21 * 7 = 147.
    
    rank = 147
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to get row index in U, V, W for a block (i, j) and offset (a, b)
    # U index: (i_block * 2 + a) * m + (j_block * 2 + b)
    # V index: (j_block * 2 + c) * p + (k_block * 2 + d) -- Wait, V is for B.
    # B is m x p. Blocks are (j, k).
    # V index: (j_block * 2 + c) * p + (k_block * 2 + d)
    # W index: (k_block * 2 + e) * n + (i_block * 2 + f)
    
    # Laderman's 21 products for 4x4 matrices (0-indexed blocks)
    # Each product is (Sum of A_blocks) * (Sum of B_blocks)
    # We define the coefficients for A and B for each of the 21 products.
    
    laderman_products = [
        # (A_indices, B_indices)
        # 1
        ([ (0,0), (3,3) ], [ (0,0), (3,3) ]),
        # 2
        ([ (0,3), (3,3) ], [ (3,0), (3,3) ]),
        # 3
        ([ (0,0), (0,1) ], [ (1,0), (0,0) ]),
        # 4
        ([ (0,0), (0,3) ], [ (3,0), (0,0) ]),
        # 5
        ([ (1,1), (3,3) ], [ (1,1), (3,3) ]),
        # 6
        ([ (1,3), (3,3) ], [ (3,1), (3,3) ]),
        # 7
        ([ (1,1), (1,2) ], [ (2,1), (1,1) ]),
        # 8
        ([ (1,1), (1,3) ], [ (3,1), (1,1) ]),
        # 9
        ([ (2,2), (3,3) ], [ (2,2), (3,3) ]),
        # 10
        ([ (2,3), (3,3) ], [ (3,2), (3,3) ]),
        # 11
        ([ (2,2), (2,1) ], [ (1,2), (2,2) ]),
        # 12
        ([ (2,2), (2,3) ], [ (3,2), (2,2) ]),
        # 13
        ([ (0,0), (1,1), (2,2), (3,3) ], [ (0,0), (1,1), (2,2), (3,3) ]),
        # 14
        ([ (0,1), (1,0), (2,3), (3,2) ], [ (0,1), (1,0), (2,3), (3,2) ]),
        # 15
        ([ (0,2), (1,3), (2,0), (3,1) ], [ (0,2), (1,3), (2,0), (3,1) ]),
        # 16
        ([ (0,3), (1,2), (2,1), (3,0) ], [ (0,3), (1,2), (2,1), (3,0) ]),
        # 17
        ([ (0,0), (0,3), (1,1), (1,2) ], [ (0,0), (0,3), (1,1), (1,2) ]),
        # 18
        ([ (0,1), (0,2), (1,0), (1,3) ], [ (0,1), (0,2), (1,0), (1,3) ]),
        # 19
        ([ (0,0), (0,1), (0,2), (0,3) ], [ (0,0), (1,0), (2,0), (3,0) ]),
        # 20
        ([ (1,0), (1,1), (1,2), (1,3) ], [ (0,1), (1,1), (2,1), (3,1) ]),
        # 21
        ([ (2,0), (2,1), (2,2), (2,3) ], [ (0,2), (1,2), (2,2), (3,2) ]),
    ]
    
    # Strassen products for 2x2 matrices
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    strassen_defs = [
        # (U_offsets, V_offsets, W_offsets, W_signs)
        # U_offsets: list of (a,b) and sign
        # V_offsets: list of (c,d) and sign
        # W_offsets: list of (e,f) and sign
        (
            [((0,0), 1), ((1,1), 1)],
            [((0,0), 1), ((1,1), 1)],
            [((0,0), 1), ((1,1), 1)]
        ),
        (
            [((1,0), 1), ((1,1), 1)],
            [((0,0), 1)],
            [((1,0), 1), ((1,1), -1)]
        ),
        (
            [((0,0), 1)],
            [((0,1), 1), ((1,1), -1)],
            [((0,1), 1), ((1,1), 1)]
        ),
        (
            [((1,1), 1)],
            [((1,0), 1), ((0,0), -1)],
            [((0,0), 1), ((1,0), 1)]
        ),
        (
            [((0,0), 1), ((0,1), 1)],
            [((1,1), 1)],
            [((0,0), -1), ((0,1), 1)]
        ),
        (
            [((1,0), 1), ((0,0), -1)],
            [((0,0), 1), ((0,1), 1)],
            [((1,1), 1)]
        ),
        (
            [((0,1), 1), ((1,1), -1)],
            [((1,0), 1), ((1,1), 1)],
            [((0,0), 1)]
        ),
    ]
    
    col = 0
    for a_indices, b_indices in laderman_products:
        for u_offsets, v_offsets, w_offsets in strassen_defs:
            # Process U
            for (bi, bj), sign in a_indices:
                for (ua, ub), usign in u_offsets:
                    # Block (bi, bj) in A
                    # Offset (ua, ub) in block
                    u_row = (bi * 2 + ua) * m + (bj * 2 + ub)
                    U[u_row, col] += sign * usign
            
            # Process V
            for (bi, bj), sign in b_indices:
                for (va, vb), vsign in v_offsets:
                    # Block (bi, bj) in B
                    # Offset (va, vb) in block
                    v_row = (bi * 2 + va) * p + (bj * 2 + vb)
                    V[v_row, col] += sign * vsign
            
            # Process W
            # W corresponds to C = AB
            # C block (ci, cj) = sum_k A(ck, dk) B(dk, cj)
            # In Laderman, the product is A_block * B_block
            # The result is added to C_block at the corresponding position?
            # No, Laderman's algorithm computes the entire 4x4 product.
            # The "product" M_k is a 2x2 matrix.
            # It contributes to the C blocks.
            # How does M_k contribute to C?
            # In Laderman, C_ij is a linear combination of the 21 products.
            # I need the coefficients for C_ij in terms of M_k.
            # I do not have these coefficients hardcoded.
            
            # This approach is flawed because I don't have the W coefficients for Laderman.
            pass
            
    # Fallback to the original 189 rank solution if I can't complete Laderman.
    # I will return the original solution.
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                def u_row(a, b):
                    return (i_block * 2 + a) * m + (k_block * 2 + b)
                def v_row(c, d):
                    return (k_block * 2 + c) * p + (j_block * 2 + d)
                def w_row(e, f):
                    return (j_block * 2 + f) * n + (i_block * 2 + e)

                # M1
                U[u_row(0, 0), col] = 1
                U[u_row(1, 1), col] = 1
                V[v_row(0, 0), col] = 1
                V[v_row(1, 1), col] = 1
                W[w_row(0, 0), col] = 1
                W[w_row(1, 1), col] = 1
                col += 1

                # M2
                U[u_row(1, 0), col] = 1
                U[u_row(1, 1), col] = 1
                V[v_row(0, 0), col] = 1
                W[w_row(1, 0), col] = 1
                W[w_row(1, 1), col] = -1
                col += 1

                # M3
                U[u_row(0, 0), col] = 1
                V[v_row(0, 1), col] = 1
                V[v_row(1, 1), col] = -1
                W[w_row(0, 1), col] = 1
                W[w_row(1, 1), col] = 1
                col += 1

                # M4
                U[u_row(1, 1), col] = 1
                V[v_row(1, 0), col] = 1
                V[v_row(0, 0), col] = -1
                W[w_row(0, 0), col] = 1
                W[w_row(1, 0), col] = 1
                col += 1

                # M5
                U[u_row(0, 0), col] = 1
                U[u_row(0, 1), col] = 1
                V[v_row(1, 1), col] = 1
                W[w_row(0, 0), col] = -1
                W[w_row(0, 1), col] = 1
                col += 1

                # M6
                U[u_row(1, 0), col] = 1
                U[u_row(0, 0), col] = -1
                V[v_row(0, 0), col] = 1
                V[v_row(0, 1), col] = 1
                W[w_row(1, 1), col] = 1
                col += 1

                # M7
                U[u_row(0, 1), col] = 1
                U[u_row(1, 1), col] = -1
                V[v_row(1, 0), col] = 1
                V[v_row(1, 1), col] = 1
                W[w_row(0, 0), col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
