"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using 3x3 blocks of 2x2 matrices
    # Each 2x2 multiplication uses Strassen's algorithm (rank 7)
    # There are 3x3x3 = 27 block multiplications
    rank = 27 * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # Block indices
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Strassen terms for 2x2 matrices
                # A block: rows 2i, 2i+1; cols 2k, 2k+1
                # B block: rows 2k, 2k+1; cols 2j, 2j+1
                # C block: rows 2i, 2i+1; cols 2j, 2j+1
                
                # Helper to add a term
                def add_term(a_indices, b_indices, c_indices, signs):
                    nonlocal column
                    for idx, sign in zip(a_indices, signs[0]):
                        U[idx][column] += sign
                    for idx, sign in zip(b_indices, signs[1]):
                        V[idx][column] += sign
                    for idx, sign in zip(c_indices, signs[2]):
                        W[idx][column] += sign
                    column += 1

                # Define indices for A, B, C elements in local 2x2 blocks
                # A: (2i, 2k), (2i, 2k+1), (2i+1, 2k), (2i+1, 2k+1)
                # B: (2k, 2j), (2k, 2j+1), (2k+1, 2j), (2k+1, 2j+1)
                # C: (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1)
                
                a00 = 2*i + 0
                a01 = 2*i + 0
                a10 = 2*i + 1
                a11 = 2*i + 1
                # Note: U is n*m. n=6, m=6.
                # Row index in U is i*m+j.
                # Here i is row block, k is col block.
                # So row index is 2*i + u, col index is 2*k + v.
                # U index = (2*i + u) * 6 + (2*k + v)
                
                # Correct U indices
                u_a00 = (2*i + 0) * 6 + (2*k + 0)
                u_a01 = (2*i + 0) * 6 + (2*k + 1)
                u_a10 = (2*i + 1) * 6 + (2*k + 0)
                u_a11 = (2*i + 1) * 6 + (2*k + 1)
                
                # V is m*p. m=6, p=6.
                # V index is j*p+k.
                # Here k is row block, j is col block.
                # So row index is 2*k + w, col index is 2*j + x.
                # V index = (2*k + w) * 6 + (2*j + x)
                
                v_b00 = (2*k + 0) * 6 + (2*j + 0)
                v_b01 = (2*k + 0) * 6 + (2*j + 1)
                v_b10 = (2*k + 1) * 6 + (2*j + 0)
                v_b11 = (2*k + 1) * 6 + (2*j + 1)
                
                # W is p*n. p=6, n=6.
                # W index is k*n+i.
                # Here i is row block, j is col block.
                # So row index is 2*j + y (col block of C becomes row index of W? No.)
                # W[k*n+i, r] is the convention.
                # The output factor follows W[k*n+i, r].
                # Usually W corresponds to C.
                # C index is (2*i + u, 2*j + v).
                # The linear index in W is (2*j + v) * 6 + (2*i + u)?
                # Or (2*i + u) * 6 + (2*j + v)?
                # The task says W[k*n+i, r].
                # If C is 6x6, and W is 36xRank.
                # The index in W corresponds to the linear index of C.
                # If C is row-major, index = row * 6 + col.
                # So W index = (2*i + u) * 6 + (2*j + v).
                
                w_c00 = (2*i + 0) * 6 + (2*j + 0)
                w_c01 = (2*i + 0) * 6 + (2*j + 1)
                w_c10 = (2*i + 1) * 6 + (2*j + 0)
                w_c11 = (2*i + 1) * 6 + (2*j + 1)
                
                # Strassen Terms
                # p1 = (a00 + a11)(b00 + b11)
                add_term([u_a00, u_a11], [v_b00, v_b11], [w_c00, w_c11], ([1, 1], [1, 1], [1, 1]))
                
                # p2 = (a10 + a11)b00
                add_term([u_a10, u_a11], [v_b00], [w_c10, w_c11], ([1, 1], [1], [1, -1]))
                
                # p3 = a00(b01 - b11)
                add_term([u_a00], [v_b01, v_b11], [w_c01, w_c11], ([1], [1, -1], [1, 1]))
                
                # p4 = a11(b10 - b00)
                add_term([u_a11], [v_b10, v_b00], [w_c00, w_c10], ([1], [1, -1], [1, 1]))
                
                # p5 = (a00 + a01)b11
                add_term([u_a00, u_a01], [v_b11], [w_c00, w_c01], ([1, 1], [1], [-1, 1]))
                
                # p6 = (a10 - a00)(b00 + b01)
                add_term([u_a10, u_a00], [v_b00, v_b01], [w_c11], ([1, -1], [1, 1], [1]))
                
                # p7 = (a01 - a11)(b10 + b11)
                add_term([u_a01, u_a11], [v_b10, v_b11], [w_c00], ([1, -1], [1, 1], [1]))
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
