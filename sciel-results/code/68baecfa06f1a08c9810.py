"""Matrix multiplication 2x3x8: rank-44 decomposition via 4 Strassen blocks + 16 schoolbook."""

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = 2, 3, 8
    # 4 Strassen blocks each handling a 2x2x2 sub-problem (using 2 of the 3 m-dimension slices)
    # Block b (b=0..3) uses m-slices 0 and 1, and p-slices [2*b, 2*b+1]
    # Plus 16 schoolbook ranks for the remaining m-slice 2 across all 8 p-slices
    num_strassen = 4 * 7  # 28
    num_schoolbook = 16   # 2 * 8
    rank = num_strassen + num_schoolbook  # 44

    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    # Strassen decomposition for 2x2x2 blocks.
    # For a block with A indices (i, j) where i in {0,1}, j in {0,1}
    # and B indices (j, k) where j in {0,1}, k in {0,1}
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    strassen_defs = [
        # (U_row, U_val, V_row, V_val, W_row, W_val) per product
        # M1: U[i*3+0]=1, U[i*3+1]=1 for i in {0,1} -> but we need per (i,j) pair
        # Actually, let me define it more carefully.
        # For block b, p-slices are pb0=2*b, pb1=2*b+1
        # m-slices are 0 and 1 (the first two of the three m-slices)
        # A is 2x2 (n=2 rows, m-slices {0,1}), B is 2x2 (m-slices {0,1}, p-slices {pb0,pb1})
    ]

    # Let me just hardcode the Strassen factors for one 2x2x2 block, then replicate.
    # For a single 2x2x2 block with m-slices {0,1} and p-slices {0,1}:
    # U has 4 rows (i*2+j for i in {0,1}, j in {0,1})
    # V has 4 rows (j*2+k for j in {0,1}, k in {0,1})
    # W has 4 rows (k*2+i for k in {0,1}, i in {0,1})

    # M1 = (A00+A11)(B00+B11):
    #   U: row 0 (i=0,j=0) += 1, row 3 (i=1,j=1) += 1
    #   V: row 0 (j=0,k=0) += 1, row 3 (j=1,k=1) += 1
    #   W: row 0 (k=0,i=0) += 1, row 3 (k=1,i=1) += 1
    # M2 = (A10+A11)B00:
    #   U: row 2 (i=1,j=0) += 1, row 3 (i=1,j=1) += 1
    #   V: row 0 (j=0,k=0) += 1
    #   W: row 2 (k=0,i=1) += 1
    # M3 = A00(B10-B00):
    #   U: row 0 (i=0,j=0) += 1
    #   V: row 2 (j=1,k=0) += 1, row 0 (j=0,k=0) -= 1
    #   W: row 1 (k=1,i=0) += 1
    # M4 = A11(B01-B11):
    #   U: row 3 (i=1,j=1) += 1
    #   V: row 1 (j=0,k=1) += 1, row 3 (j=1,k=1) -= 1
    #   W: row 3 (k=1,i=1) += 1
    # M5 = (A00+A10)B11:
    #   U: row 0 (i=0,j=0) += 1, row 2 (i=1,j=0) += 1
    #   V: row 3 (j=1,k=1) += 1
    #   W: row 0 (k=0,i=0) -= 1, row 1 (k=1,i=0) += 1
    # M6 = (A11-A10)(B00+B10):
    #   U: row 3 (i=1,j=1) += 1, row 2 (i=1,j=0) -= 1
    #   V: row 0 (j=0,k=0) += 1, row 2 (j=1,k=0) += 1
    #   W: row 3 (k=1,i=1) += 1, row 2 (k=0,i=1) += 1
    # M7 = (A01-A11)(B01+B11):
    #   U: row 1 (i=0,j=1) += 1, row 3 (i=1,j=1) -= 1
    #   V: row 1 (j=0,k=1) += 1, row 3 (j=1,k=1) += 1
    #   W: row 0 (k=0,i=0) -= 1, row 1 (k=1,i=0) += 1

    strassen_products = [
        # (U_offsets, V_offsets, W_offsets)
        # Each offset is (row, val)
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3
        ([(0, 1)], [(2, 1), (0, -1)], [(1, 1)]),
        # M4
        ([(3, 1)], [(1, 1), (3, -1)], [(3, 1)]),
        # M5
        ([(0, 1), (2, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(3, 1), (2, -1)], [(0, 1), (2, 1)], [(3, 1), (2, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(1, 1), (3, 1)], [(0, -1), (1, 1)]),
    ]

    for b in range(4):
        base = b * 7
        pb0 = 2 * b
        pb1 = 2 * b + 1
        for prod_idx in range(7):
            col = base + prod_idx
            u_offsets, v_offsets, w_offsets = strassen_products[prod_idx]
            for (local_row, val) in u_offsets:
                # local_row is in {0,1,2,3} corresponding to (i,j) in 2x2
                i = local_row // 2
                j = local_row % 2
                U[i * m + j][col] += val
            for (local_row, val) in v_offsets:
                # local_row is in {0,1,2,3} corresponding to (j,k) in 2x2
                j = local_row // 2
                k_local = local_row % 2
                k = pb0 + k_local
                V[j * p + k][col] += val
            for (local_row, val) in w_offsets:
                # local_row is in {0,1,2,3} corresponding to (k,i) in 2x2
                k_local = local_row // 2
                i = local_row % 2
                k = pb0 + k_local
                W[k * n + i][col] += val

    # Schoolbook for m-slice 2, all p-slices 0..7
    col = num_strassen
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
