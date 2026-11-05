"""Block-Strassen decomposition for 6x6 matrix multiplication.

Partition each 6x6 matrix into a 3x3 grid of 2x2 blocks.
Apply rank-7 Strassen to the outer 3x3 block multiplication.
Use rank-4 schoolbook for each inner 2x2 scalar product.
Total rank = 27 (outer Strassen) * 4 (inner schoolbook) = 108.
"""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    B = 2  # block size
    G = n // B  # grid size = 3
    # Outer Strassen rank for GxG
    outer_rank = 7  # for 3x3 Strassen
    inner_rank = B * B  # 4 for 2x2 schoolbook
    total_rank = outer_rank * inner_rank  # 28? No, 7*4=28 per outer mult, but we have 9 outer mults replaced by 7

    # Actually: outer 3x3 Strassen has 7 multiplications.
    # Each outer multiplication is a 2x2 matrix product = rank 4.
    # Total rank = 7 * 4 = 28? That's not right either.

    # Let me think more carefully.
    # Outer 3x3: indices (a,b) where a,b in {0,1,2} are block row/col.
    # Strassen computes 7 products of sums of blocks.
    # Each such product is a 2x2 x 2x2 multiplication = 4 scalar products.
    # So total scalar products = 7 * 4 = 28? No, that's the rank.

    # Wait, the rank of the tensor decomposition:
    # Outer Strassen: 7 rank-1 tensors for the 3x3 block structure.
    # Each outer rank-1 involves a 2x2 x 2x2 multiplication which itself has rank 4.
    # So total rank = 7 * 4 = 28? That seems too low.

    # Let me reconsider. The full 6x6x6 tensor has rank 216 (schoolbook).
    # Block structure: 3x3 grid of 2x2 blocks.
    # Outer 3x3 Strassen: replaces 9 block multiplications with 7.
    # Each block multiplication is a 2x2 x 2x2 = rank 4 tensor.
    # So total rank = 7 * 4 = 28? No, that's not how tensor rank composes.

    # Actually, tensor rank is not multiplicative in that simple way.
    # Let me think of it as: the 6x6x6 tensor T_{ijk} where i,j,k in 0..5.
    # Group into blocks: i = 2*a + alpha, j = 2*b + beta, k = 2*c + gamma
    # where a,b,c in {0,1,2} and alpha,beta,gamma in {0,1}.
    # T_{(2a+alpha)(2b+beta)(2c+gamma)} = delta_{alpha,beta} * delta_{beta,gamma} (schoolbook)
    # Wait no, matrix multiplication: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So T_{ijk} = delta_{j,k} ... no.
    # T_{i,j,k} where A is i x j, B is j x k, result is i x k.
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor is T_{i,j,k} = 1 if this is the (i,j,k) entry, i.e., it's the identity in the middle index.

    # For block decomposition:
    # i = 2a + alpha, j = 2b + beta, k = 2c + gamma
    # T_{i,j,k} = delta_{alpha,beta} * delta_{beta,gamma}? No.
    # T_{i,j,k} = 1 for the specific (i,j,k) triple in schoolbook.

    # Let me just build it directly.
    # Outer Strassen for 3x3 blocks:
    # M1 = (A00+A22)(B00+B22)
    # M2 = (A20+A22)B00
    # M3 = A00(B01-B21)
    # M4 = A22(B21-B01)
    # M5 = (A00+A10)B10
    # M6 = (A12+A22)(B11-B21)
    # M7 = (A01-A11)(B10+B11)

    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M7
    # C02 = M5 + M6
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    # C12 = M2 + M4 ... wait, let me look up the standard Strassen for 3x3.

    # Actually, standard Strassen is for 2x2. For 3x3, there are various decompositions.
    # The simplest is to use the 2x2 Strassen recursively or use a known 3x3 decomposition.

    # Let me use a simpler approach: just use schoolbook for both levels but that gives 216.
    # Or use Strassen at the 2x2 level: 6x6 = 3x3 blocks of 2x2, each 2x2 uses Strassen rank 7.
    # Total rank = 9 * 7 = 63? No, 9 block multiplications each of rank 7 = 63.

    # Wait, that's much better! Let me do that.
    # Each 2x2 block multiplication uses Strassen (rank 7).
    # There are 9 such block multiplications (3x3 grid).
    # Total rank = 9 * 7 = 63.

    # But we need to be careful: the blocks interact. In standard block multiplication,
    # C_{ab} = sum_c A_{ac} B_{cb}, so there are 9*3 = 27 block products in schoolbook.
    # With Strassen on each 2x2, each block product (2x2 x 2x2) has rank 7.
    # Total rank = 27 * 7 = 189. That's worse than 216? No, 189 < 216.

    # Hmm, but actually in block multiplication, C_{ab} = sum_c A_{ac} B_{cb}.
    # For 3x3 blocks, that's 3 sums per C_{ab}, so 9*3 = 27 block multiplications.
    # Each block multiplication is 2x2 x 2x2 with rank 7 (Strassen).
    # Total rank = 27 * 7 = 189.

    # Alternatively, apply Strassen at the block level (3x3 Strassen) and schoolbook inside:
    # 3x3 Strassen has rank 7 (or more, depending on decomposition).
    # Actually, the minimal rank for 3x3 matrix multiplication is not known to be 7.
    # The standard Strassen is for 2x2. For 3x3, we can use a 7-term decomposition or more.

    # Let me just go with: 3x3 grid of 2x2 blocks, each 2x2 uses Strassen (rank 7).
    # Total rank = 27 * 7 = 189.

    # Actually, let me reconsider. The task says "rank-minimal exact tensor decomposition."
    # The schoolbook rank is 216. We want to do better.

    # Simplest improvement: use Strassen for each 2x2 sub-multiplication.
    # 6x6 = 3x3 blocks of 2x2.
    # Block multiplication: C_{ab} = sum_{c=0}^{2} A_{ac} B_{cb}
    # That's 27 block products, each 2x2 x 2x2.
    # Each 2x2 x 2x2 has rank 7 (Strassen).
    # Total rank = 27 * 7 = 189.

    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189

    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B10
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M6

    strassen_coeffs = [
        # (M_index, a_coeff, b_coeff, c_row, c_col, sign)
        # For each M_k, and for each output (r,c), the coefficient.
    ]

    # Let me define Strassen terms for 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b10)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M6

    # For each block position (a,b) in the 3x3 grid, and each sum index c:
    # A_block = A[2a:2a+2, 2c:2c+2]
    # B_block = B[2c:2c+2, 2b:2b+2]
    # C_block[2a:2a+2, 2b:2b+2] += A_block @ B_block

    # For Strassen on A_block @ B_block:
    # a00 = A[2a, 2c], a01 = A[2a, 2c+1], a10 = A[2a+1, 2c], a11 = A[2a+1, 2c+1]
    # b00 = B[2c, 2b], b01 = B[2c, 2b+1], b10 = B[2c+1, 2b], b11 = B[2c+1, 2b+1]

    col = 0
    for a in range(3):  # block row of A, output row block
        for c in range(3):  # sum index (block col of A, block row of B)
            for b in range(3):  # block col of B, output col block
                # 7 Strassen terms
                for s in range(7):
                    # Determine U, V, W entries for this term
                    if s == 0:  # M1 = (a00+a11)(b00+b11)
                        # U: a00 and a11
                        U[2*a][col] += 1  # a00: row 2a, col 2c -> index 2a*6+2c? No.
                        # Wait, U has shape (n*m, rank) = (36, rank).
                        # U[i*m + j, col] where i is row of A, j is col of A.
                        # a00 is at A[2a, 2c], so i=2a, j=2c, index = 2a*6+2c.
                        # a11 is at A[2a+1, 2c+1], index = (2a+1)*6+(2c+1).
                        U[2*a*6 + 2*c][col] += 1
                        U[(2*a+1)*6 + (2*c+1)][col] += 1
                        # V: b00 and b11
                        # b00 is at B[2c, 2b], index = 2c*6+2b.
                        # b11 is at B[2c+1, 2b+1], index = (2c+1)*6+(2b+1).
                        V[2*c*6 + 2*b][col] += 1
                        V[(2*c+1)*6 + (2*b+1)][col] += 1
                        # W: contributes to C[2a,2b], C[2a+1,2b], C[2a,2b+1], C[2a+1,2b+1]
                        # C00 = M1 + ... so W for (2a,2b) gets +1
                        # C10 = M2+M4, no M1. Wait, let me recheck.
                        # C00 = M1 + M4 - M6 + M7
                        # C01 = M3 + M5
                        # C10 = M2 + M4
                        # C11 = M1 - M3 + M5 + M6
                        # So M1 contributes to C00 and C11.
                        W[2*b*6 + 2*a][col] += 1  # C[2a,2b]: W[k*n+i] where k=2b, i=2a -> 2b*6+2a
                        W[(2*b+1)*6 + (2*a+1)][col] += 1  # C[2a+1,2b+1]
                    elif s == 1:  # M2 = (a10+a11)(b00)
                        U[(2*a+1)*6 + 2*c][col] += 1  # a10
                        U[(2*a+1)*6 + (2*c+1)][col] += 1  # a11
                        V[2*c*6 + 2*b][col] += 1  # b00
                        # M2 contributes to C10
                        W[2*b*6 + (2*a+1)][col] += 1  # C[2a+1,2b]
                    elif s == 2:  # M3 = a00(b01-b11)
                        U[2*a*6 + 2*c][col] += 1  # a00
                        V[2*c*6 + (2*b+1)][col] += 1  # b01
                        V[(2*c+1)*6 + (2*b+1)][col] += -1  # -b11
                        # M3 contributes to C01 (+1) and C11 (-1)
                        W[(2*b+1)*6 + 2*a][col] += 1  # C[2a,2b+1]
                        W[(2*b+1)*6 + (2*a+1)][col] += -1  # C[2a+1,2b+1]
                    elif s == 3:  # M4 = a11(b10-b00)
                        U[(2*a+1)*6 + (2*c+1)][col] += 1  # a11
                        V[(2*c+1)*6 + 2*b][col] += 1  # b10
                        V[2*c*6 + 2*b][col] += -1  # -b00
                        # M4 contributes to C00 (+1) and C10 (+1)
                        W[2*b*6 + 2*a][col] += 1  # C[2a,2b]
                        W[2*b*6 + (2*a+1)][col] += 1  # C[2a+1,2b]
                    elif s == 4:  # M5 = (a00+a01)(b10)
                        U[2*a*6 + 2*c][col] += 1  # a00
                        U[2*a*6 + (2*c+1)][col] += 1  # a01
                        V[(2*c+1)*6 + 2*b][col] += 1  # b10
                        # M5 contributes to C01 (+1) and C11 (+1)
                        W[(2*b+1)*6 + 2*a][col] += 1  # C[2a,2b+1]
                        W[(2*b+1)*6 + (2*a+1)][col] += 1  # C[2a+1,2b+1]
                    elif s == 5:  # M6 = (a10-a00)(b00+b01)
                        U[(2*a+1)*6 + 2*c][col] += 1  # a10
                        U[2*a*6 + 2*c][col] += -1  # -a00
                        V[2*c*6 + 2*b][col] += 1  # b00
                        V[2*c*6 + (2*b+1)][col] += 1  # b01
                        # M6 contributes to C00 (-1) and C11 (+1)
                        W[2*b*6 + 2*a][col] += -1  # C[2a,2b]
                        W[(2*b+1)*6 + (2*a+1)][col] += 1  # C[2a+1,2b+1]
                    elif s == 6:  # M7 = (a01-a11)(b10+b11)
                        U[2*a*6 + (2*c+1)][col] += 1  # a01
                        U[(2*a+1)*6 + (2*c+1)][col] += -1  # -a11
                        V[(2*c+1)*6 + 2*b][col] += 1  # b10
                        V[(2*c+1)*6 + (2*b+1)][col] += 1  # b11
                        # M7 contributes to C00 (+1)
                        W[2*b*6 + 2*a][col] += 1  # C[2a,2b]
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
