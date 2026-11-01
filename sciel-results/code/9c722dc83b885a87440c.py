"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (10, rank)
    # W: (p*n, rank) = (10, rank)
    # Index conventions:
    #   U row: i*m + j  (i in 0..n-1, j in 0..m-1)
    #   V row: j*p + k  (j in 0..m-1, k in 0..p-1)
    #   W row: k*n + i  (k in 0..p-1, i in 0..n-1)

    # We use a hybrid decomposition:
    #   - For k-blocks {0,1} and {2,3}: apply Strassen to 2x2 * 2x2 (rank 7 each)
    #   - For k-block {4}: schoolbook 2x2 * 2x1 (rank 4)
    # Total rank = 7 + 7 + 4 = 18

    rank = 18
    U = [[0] * rank for _ in range(n * m)]   # 4 x 18
    V = [[0] * rank for _ in range(m * p)]   # 10 x 18
    W = [[0] * rank for _ in range(p * n)]   # 10 x 18

    # Strassen for 2x2 * 2x2:
    # Given A (2x2), B (2x2), C = A*B
    # Partition A = [[a11, a12],[a21, a22]], B = [[b11, b12],[b21, b22]]
    # C = [[c11, c12],[c21, c22]]
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22) b11
    # M3 = a11 (b12-b22)
    # M4 = a22 (b21-b11)
    # M5 = (a11+a12) b22
    # M6 = (a21-a11) (b11+b12)
    # M7 = (a12-a22) (b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # Each Mj is a rank-1 term: (linear combo of A) * (linear combo of B) -> (linear combo of C)
    # We need to express each as U_col * V_col * W_col where:
    #   U_col[i*m+j] = coeff of a_{ij}
    #   V_col[j*p+k] = coeff of b_{jk}
    #   W_col[k*n+i] = coeff of c_{ik}

    def add_strassen_block(k_offset, col_start):
        """Add Strassen decomposition for k-block [k_offset, k_offset+1]."""
        # Local indices within this 2x2 block:
        # A indices: a_{00}=0, a_{01}=1, a_{10}=2, a_{11}=3  (global i*m+j)
        # B local: b_{00}=0, b_{01}=1, b_{10}=2, b_{11}=3
        #   Global B index: j*p + (k_offset + k_local)
        # C local: c_{00}=0, c_{01}=1, c_{10}=2, c_{11}=3
        #   Global C index: (k_offset + k_local)*n + i

        # Strassen factors (u, v, w) where u is 4-vector (a coeffs), v is 4-vector (b local coeffs), w is 4-vector (c local coeffs)
        factors = [
            # M1 = (a11+a22)(b11+b22)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
            # M2 = (a21+a22) b11
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, -1, 1]),
            # M3 = a11 (b12-b22)
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
            # M4 = a22 (b21-b11)
            ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 1]),
            # M5 = (a11+a12) b22
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),
            # M6 = (a21-a11) (b11+b12)
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7 = (a12-a22) (b21+b22)
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]

        for (u, v, w) in factors:
            col = col_start
            # Fill U column
            for idx in range(4):
                U[idx][col] = u[idx]
            # Fill V column: local b index l -> global j*p + (k_offset + k_l)
            # local b: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
            b_map = [(0, k_offset), (0, k_offset + 1), (1, k_offset), (1, k_offset + 1)]
            for l, (j, k) in enumerate(b_map):
                V[j * p + k][col] = v[l]
            # Fill W column: local c index l -> global (k_offset + k_l)*n + i
            # local c: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
            c_map = [(k_offset, 0), (k_offset, 1), (k_offset + 1, 0), (k_offset + 1, 1)]
            for l, (k, i) in enumerate(c_map):
                W[k * n + i][col] = w[l]
            col_start += 1
        return col_start

    # Block 1: k in {0, 1}
    col = add_strassen_block(0, 0)  # uses cols 0..6

    # Block 2: k in {2, 3}
    col = add_strassen_block(2, col)  # uses cols 7..13

    # Block 3: k = 4, schoolbook 2x2 * 2x1 (rank 4)
    # C_{i,4} = sum_j a_{ij} * b_{j,4}
    # 4 terms:
    #   a_{00} b_{04} -> c_{04}: U[0], V[0*5+4], W[4*2+0]
    #   a_{01} b_{14} -> c_{04}: U[1], V[1*5+4], W[4*2+0]
    #   a_{10} b_{04} -> c_{14}: U[2], V[0*5+4], W[4*2+1]
    #   a_{11} b_{14} -> c_{14}: U[3], V[1*5+4], W[4*2+1]
    terms = [
        (0, 0 * p + 4, 4 * n + 0),
        (1, 1 * p + 4, 4 * n + 0),
        (2, 0 * p + 4, 4 * n + 1),
        (3, 1 * p + 4, 4 * n + 1),
    ]
    for (u_idx, v_idx, w_idx) in terms:
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
