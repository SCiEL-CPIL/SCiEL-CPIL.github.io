"""Matrix multiplication 6x6x6 via Strassen-on-2x2-blocks with schoolbook 3x3 blocks.

Rank = 7 * 27 = 189 (Strassen 2x2 block decomposition, each block product done schoolbook).
Output factor follows the official trace convention W[k*n + i, c].
"""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: build is trivial and fast, but keep a safety margin.
    t0 = time.time()

    n, m, p = DIMENSIONS  # all 6
    # 2x2 block grid: each block is 3x3.
    B = 3

    # Strassen 2x2 block decomposition.
    # For 2x2 matrices A, B (each 2x2 blocks) and product C = A*B,
    # define the 7 products:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    # and C blocks:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # We build the tensor factors. For each of the 7 "multiplications" we
    # need the input (U, V) linear combinations and the output (W) linear
    # combination. Each block is 3x3, so each block product is itself a
    # schoolbook 3x3x3 tensor (rank 27).

    # Represent each block by its (row_block, col_block) in the 2x2 grid.
    # A block (rb, cb) covers rows rb*B..(rb+1)*B-1 and cols cb*B..(cb+1)*B-1.

    # U factors act on A: index (i, j) -> i*m + j, where i is global row, j global col.
    # For a block (rb, cb) of A, local (a, b) maps to global i = rb*B + a, j = cb*B + b.
    # V factors act on B: index (j, k) -> j*p + k, where j is global row of B (== global col of A),
    # k is global col of B.
    # For a block (rb, cb) of B, local (a, b) maps to global j = rb*B + a, k = cb*B + b.
    # W factors act on C: index (k, i) -> k*n + i, where k is global row of C, i is global col of C.
    # For a block (rb, cb) of C, local (a, b) maps to global k = rb*B + a, i = cb*B + b.

    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def u_idx(rb, cb, a, b):
        i = rb * B + a
        j = cb * B + b
        return i * m + j

    def v_idx(rb, cb, a, b):
        j = rb * B + a
        k = cb * B + b
        return j * p + k

    def w_idx(rb, cb, a, b):
        k = rb * B + a
        i = cb * B + b
        return k * n + i

    # Each Strassen multiplication: (A_block_combo, B_block_combo, [C_block_combo terms])
    # A_block_combo: list of (rb, cb, coeff) for A blocks
    # B_block_combo: list of (rb, cb, coeff) for B blocks
    # C_block_combo: list of (rb, cb, coeff) for C blocks (with signs)

    strassen_mults = [
        # M1 = (A00 + A11) * (B00 + B11)
        ([((0, 0, 1), (1, 1, 1))],
         [((0, 0, 1), (1, 1, 1))],
         [((0, 0, 1), (1, 1, 1))]),
        # M2 = (A10 + A11) * B00
        ([((1, 0, 1), (1, 1, 1))],
         [((0, 0, 1))],
         [((1, 0, 1), (1, 1, -1))]),
        # M3 = A00 * (B01 - B11)
        ([((0, 0, 1))],
         [((0, 1, 1), (1, 1, -1))],
         [((0, 1, 1), (1, 1, 1))]),
        # M4 = A11 * (B10 - B00)
        ([((1, 1, 1))],
         [((1, 0, 1), (0, 0, -1))],
         [((0, 0, 1), (1, 0, 1))]),
        # M5 = (A00 + A01) * B11
        ([((0, 0, 1), (0, 1, 1))],
         [((1, 1, 1))],
         [((0, 0, -1), (1, 1, 1))]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([((1, 0, 1), (0, 0, -1))],
         [((0, 0, 1), (0, 1, 1))],
         [((1, 1, 1))]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([((0, 1, 1), (1, 1, -1))],
         [((1, 0, 1), (1, 1, 1))],
         [((0, 0, 1))]),
    ]

    col = 0
    for (a_combo, b_combo, c_combo) in strassen_mults:
        # a_combo is a list of (rb, cb, coeff); we need all A blocks involved.
        # For each A block in a_combo and each B block in b_combo, and for each
        # local (a, b) in the 3x3 block, we create one rank-1 term (schoolbook).
        # U coefficient = A_coeff, V coefficient = B_coeff, W coefficient = sum over C blocks of C_coeff.
        for (ar, ac, a_coeff) in a_combo:
            for (br, bc, b_coeff) in b_combo:
                for a in range(B):
                    for b in range(B):
                        # W coefficient: sum of C coeffs for the output block.
                        # The output block for a product of A block (ar, ac) and B block (br, bc)
                        # is C block (ar, bc).
                        w_coeff = 0.0
                        for (cr, cc, c_coeff) in c_combo:
                            if cr == ar and cc == bc:
                                w_coeff += c_coeff
                        if w_coeff == 0:
                            continue
                        ui = u_idx(ar, ac, a, b)
                        vi = v_idx(br, bc, a, b)
                        wi = w_idx(ar, bc, a, b)
                        U[ui][col] = a_coeff
                        V[vi][col] = b_coeff
                        W[wi][col] = w_coeff
                        col += 1

    assert col == rank, f"col={col}, rank={rank}"

    if time.time() - t0 > 100:
        # Should never happen; return what we have.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
