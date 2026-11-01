"""
Rank-49 tensor decomposition for 4x4 matrix multiplication.

This implements the recursive Strassen construction:
- Split 4x4 into 2x2 blocks (4 blocks per matrix)
- Apply Strassen's 7-multiplication formula at the block level
- Apply Strassen's 7-multiplication formula again at the element level within each 2x2

This gives rank = 7 * 7 = 49, which is the known minimum for 4x4 matrix multiplication.

The output factor W follows the official trace convention W[k*n+i, r].
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # all 4
    rank = 49

    # We'll build U (16 x 49), V (16 x 49), W (16 x 49)
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49

    # Strassen 2x2 formula:
    # For matrices A, B (2x2), the product C = A*B is computed with 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Strassen coefficients for U (A side):
    # M1: A00=1, A11=1
    # M2: A10=1, A11=1
    # M3: A00=1
    # M4: A11=1
    # M5: A00=1, A01=1
    # M6: A10=1, A00=-1
    # M7: A01=1, A11=-1

    # Strassen coefficients for V (B side):
    # M1: B00=1, B11=1
    # M2: B00=1
    # M3: B01=1, B11=-1
    # M4: B10=1, B00=-1
    # M5: B11=1
    # M6: B00=1, B01=1
    # M7: B10=1, B11=1

    # Strassen coefficients for W (C side):
    # M1: C00=1, C11=1
    # M2: C10=1, C11=-1
    # M3: C01=1, C11=1
    # M4: C00=1, C10=1
    # M5: C00=-1, C01=1
    # M6: C11=1
    # M7: C00=1, C11=1

    # Indexing for 2x2 blocks:
    # A is 4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # A00 covers rows 0-1, cols 0-1
    # A01 covers rows 0-1, cols 2-3
    # A10 covers rows 2-3, cols 0-1
    # A11 covers rows 2-3, cols 2-3

    # For the 4x4 matrix A (16 elements), element (i,j) where i,j in 0..3:
    # Block row = i // 2, block col = j // 2
    # Within block: row = i % 2, col = j % 2

    # Strassen inner formula for 2x2 matrices:
    # The 7 inner products for a 2x2 multiplication:
    # inner_U[prod_idx][elem_idx] where elem_idx is 0..3 for the 2x2 A
    #   elem 0 = A[0,0], elem 1 = A[0,1], elem 2 = A[1,0], elem 3 = A[1,1]
    inner_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]

    inner_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [0, -1, 1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]

    inner_W = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 1],   # M7: C00 + C11
    ]

    # Outer Strassen formula for 2x2 blocks of 2x2 matrices:
    # The 4x4 A is split into 4 blocks (A00, A01, A10, A11), each 2x2.
    # The outer Strassen uses the same 7-product structure but on blocks.
    #
    # outer_U[prod_idx][block_idx] where block_idx is 0..3 for the 4 blocks
    #   block 0 = A00, block 1 = A01, block 2 = A10, block 3 = A11
    outer_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]

    outer_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [0, -1, 1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]

    outer_W = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 1],   # M7: C00 + C11
    ]

    # Now build the full 49-rank decomposition.
    # Each of the 49 columns corresponds to (outer_prod, inner_prod) where
    # outer_prod in 0..6, inner_prod in 0..6.
    #
    # For element (i, j) of the 4x4 matrix A:
    #   block_row = i // 2, block_col = j // 2
    #   outer_block = block_row * 2 + block_col  (0=A00, 1=A01, 2=A10, 3=A11)
    #   inner_row = i % 2, inner_col = j % 2
    #   inner_elem = inner_row * 2 + inner_col (0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1])
    #
    # U[i*m+j, col] = outer_U[outer_prod][outer_block] * inner_U[inner_prod][inner_elem]
    # V[j*p+k, col] = outer_V[outer_prod][outer_block_B] * inner_V[inner_prod][inner_elem_B]
    # W[k*n+i, col] = outer_W[outer_prod][outer_block_C] * inner_W[inner_prod][inner_elem_C]
    #
    # For A (input, n x m = 4x4): element (i, j)
    #   outer_block_A = (i // 2) * 2 + (j // 2)
    #   inner_elem_A = (i % 2) * 2 + (j % 2)
    #
    # For B (input, m x p = 4x4): element (j, k)
    #   outer_block_B = (j // 2) * 2 + (k // 2)
    #   inner_elem_B = (j % 2) * 2 + (k % 2)
    #
    # For C (output, p x n = 4x4): element (k, i)  [W uses k*n+i indexing]
    #   outer_block_C = (k // 2) * 2 + (i // 2)
    #   inner_elem_C = (k % 2) * 2 + (i % 2)

    for outer_prod in range(7):
        for inner_prod in range(7):
            col = outer_prod * 7 + inner_prod
            for i in range(n):
                for j in range(m):
                    # A element (i, j)
                    outer_block_A = (i // 2) * 2 + (j // 2)
                    inner_elem_A = (i % 2) * 2 + (j % 2)
                    U[i * m + j][col] = outer_U[outer_prod][outer_block_A] * inner_U[inner_prod][inner_elem_A]

                    # B element (j, k) - but we need to iterate over k for V
                    # Actually V is indexed by (j, k) for B which is m x p
                    # Let's handle V and W in separate loops

            for j in range(m):
                for k in range(p):
                    # B element (j, k)
                    outer_block_B = (j // 2) * 2 + (k // 2)
                    inner_elem_B = (j % 2) * 2 + (k % 2)
                    V[j * p + k][col] = outer_V[outer_prod][outer_block_B] * inner_V[inner_prod][inner_elem_B]

                    # C element (k, i) - but i is not in this loop scope properly
                    # Let me restructure

    # Let me redo this more carefully with a single triple loop
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for outer_prod in range(7):
        for inner_prod in range(7):
            col = outer_prod * 7 + inner_prod
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # A element (i, j)
                        outer_block_A = (i // 2) * 2 + (j // 2)
                        inner_elem_A = (i % 2) * 2 + (j % 2)
                        U[i * m + j][col] = outer_U[outer_prod][outer_block_A] * inner_U[inner_prod][inner_elem_A]

                        # B element (j, k)
                        outer_block_B = (j // 2) * 2 + (k // 2)
                        inner_elem_B = (j % 2) * 2 + (k % 2)
                        V[j * p + k][col] = outer_V[outer_prod][outer_block_B] * inner_V[inner_prod][inner_elem_B]

                        # C element (k, i) - W is indexed as W[k*n + i, col]
                        outer_block_C = (k // 2) * 2 + (i // 2)
                        inner_elem_C = (k % 2) * 2 + (i % 2)
                        W[k * n + i][col] = outer_W[outer_prod][outer_block_C] * inner_W[inner_prod][inner_elem_C]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
