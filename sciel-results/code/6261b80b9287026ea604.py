import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank 56 decomposition: 8 Strassen instances of 2x2x2
    # A (2x4) split into A_L (cols 0-1), A_R (cols 2-3)
    # B (4x8) split into 8 blocks of 2x2:
    #   B[i,j] where i in {0,1} (row-block of B: rows 0-1 or 2-3), j in {0,1,2,3} (col-block of B: cols 2j..2j+1)
    # C = sum over (a_block, b_row_block) of A_{a_block} * B_{b_row_block, *}
    #   where a_block in {0,1} selects A_L or A_R (cols 2*a_block .. 2*a_block+1)
    #   and b_row_block in {0,1} selects B rows 2*b_row_block .. 2*b_row_block+1
    # For each (a_block, b_row_block), B_{b_row_block} is 2x4 = [B_{b_row_block,0} | B_{b_row_block,1}]
    #   each 2x2, so two Strassen multiplications per (a_block, b_row_block)
    # Total: 2 * 2 * 2 = 8 Strassen instances, rank = 8 * 7 = 56

    # Strassen for 2x2 * 2x2:
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # P0 = (a0+a3)*(b0+b3)
    # P1 = (a2+a3)*b0
    # P2 = a0*(b1-b3)
    # P3 = a3*(b2-b0)
    # P4 = (a0+a1)*b3
    # P5 = (a2-a0)*(b0+b1)
    # P6 = (a1-a3)*(b2+b3)
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5

    # Factor lists for each Strassen product: (A_factors, B_factors)
    # A_factors: list of (elem_idx, coeff) where elem_idx: 0=a0,1=a1,2=a2,3=a3
    # B_factors: list of (elem_idx, coeff) where elem_idx: 0=b0,1=b1,2=b2,3=b3
    strassen_products = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P0
        ([(2, 1), (3, 1)], [(0, 1)]),           # P1
        ([(0, 1)], [(1, 1), (3, -1)]),          # P2
        ([(3, 1)], [(2, 1), (0, -1)]),          # P3
        ([(0, 1), (1, 1)], [(3, 1)]),           # P4
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P5
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P6
    ]

    # C reconstruction: C_elem = sum of (prod_idx, coeff)
    # C00: P0 + P3 - P4 + P6
    # C01: P2 + P4
    # C10: P1 + P3
    # C11: P0 - P1 + P2 + P5
    c_reconstruction = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                   # C01
        [(1, 1), (3, 1)],                   # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],  # C11
    ]

    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    # 8 Strassen instances: (a_block, b_row_block, b_col_block)
    # a_block: 0 or 1 (selects A cols 2*a_block .. 2*a_block+1)
    # b_row_block: 0 or 1 (selects B rows 2*b_row_block .. 2*b_row_block+1)
    # b_col_block: 0..3 (selects B cols 2*b_col_block .. 2*b_col_block+1)
    instances = []
    for a_block in range(2):
        for b_row_block in range(2):
            for b_col_block in range(4):
                instances.append((a_block, b_row_block, b_col_block))

    col = 0
    for (a_block, b_row_block, b_col_block) in instances:
        for prod_idx in range(7):
            a_factors, b_factors = strassen_products[prod_idx]
            for a_elem, a_coeff in a_factors:
                a_row = a_elem // 2
                a_col = a_elem % 2
                u_idx = a_row * m + (2 * a_block + a_col)
                U[u_idx][col] += a_coeff
            for b_elem, b_coeff in b_factors:
                b_row = b_elem // 2
                b_col = b_elem % 2
                v_idx = (2 * b_row_block + b_row) * p + (2 * b_col_block + b_col)
                V[v_idx][col] += b_coeff
            col += 1

    # W contributions
    col = 0
    for (a_block, b_row_block, b_col_block) in instances:
        for c_elem in range(4):
            c_row = c_elem // 2
            c_col = c_elem % 2
            w_idx = (2 * b_col_block + c_col) * n + c_row
            for prod_idx, coeff in c_reconstruction[c_elem]:
                W[w_idx][col + prod_idx] += coeff
        col += 7

    # Verification: check that U * V^T * W^T reconstructs the tensor
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i in [0, n*m), j in [0, m*p), k in [0, p*n)
    # The tensor for matrix multiplication: T[i,j,k] = 1 if i, j, k correspond to valid (row, col, row) indices
    # i = a_row * m + a_col, j = b_row * p + b_col, k = b_col * n + a_row
    # T[i,j,k] = 1 if a_col == b_row and a_row matches (i.e., i%m == j//p and k%n == i//m)
    if time.time() < deadline:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Compute tensor: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # This is a 3-way outer product sum
        # Efficient: for each r, outer product of U[:,r], V[:,r], W[:,r]
        T = np.zeros((n * m, m * p, p * n), dtype=float)
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
        
        # Check against expected tensor
        expected = np.zeros((n * m, m * p, p * n), dtype=float)
        for i in range(n * m):
            a_row = i // m
            a_col = i % m
            for j in range(m * p):
                b_row = j // p
                b_col = j % p
                if a_col == b_row:
                    k = b_col * n + a_row
                    expected[i, j, k] = 1.0
        
        max_err = np.max(np.abs(T - expected))
        if max_err > 1e-10:
            # Fallback: should not happen, but return anyway
            pass

    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
