"""Rewritten program for 3x4x8 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Initialize U, V, W as sparse lists of (index, value) pairs
    U = [[] for _ in range(n * m)]
    V = [[] for _ in range(m * p)]
    W = [[] for _ in range(p * n)]

    rank = 0

    def add_term(u_entries, v_entries, w_entries):
        nonlocal rank
        U.append(u_entries)
        V.append(v_entries)
        W.append(w_entries)
        rank += 1

    # Schoolbook for the remaining part: i=2 (bottom row), all j,k
    # This covers A[2,j] * B[j,k] -> C[k,2]
    # U index: 2*m + j = 8 + j
    # V index: j*p + k
    # W index: k*n + 2 = 3k + 2
    for j in range(m):
        for k in range(p):
            add_term([(8 + j, 1)], [(j * p + k, 1)], [(k * n + 2, 1)])

    # Strassen for the top 2x2 blocks: i in {0,1}, split j into pairs, split k into pairs
    # Strassen for multiplying A_{2x2} (rows i0,i0+1, cols j0,j0+1) by B_{2x2} (rows j0,j0+1, cols k0,k0+1)
    # Result: C_{2x2} (rows i0,i0+1, cols k0,k0+1)
    #
    # Strassen formulas:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A10) * B11
    # M6 = (A11-A10) * (B00+B10)
    # M7 = (A01-A11) * (B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Where A00 = A[i0, j0], A01 = A[i0, j0+1], A10 = A[i0+1, j0], A11 = A[i0+1, j0+1]
    #       B00 = B[j0, k0], B01 = B[j0, k0+1], B10 = B[j0+1, k0], B11 = B[j0+1, k0+1]
    #       C00 = C[i0, k0], C01 = C[i0, k0+1], C10 = C[i0+1, k0], C11 = C[i0+1, k0+1]

    def strassen_block(i0, j0, k0):
        # U indices for A entries: i*m + j
        u00 = i0 * m + j0
        u01 = i0 * m + (j0 + 1)
        u10 = (i0 + 1) * m + j0
        u11 = (i0 + 1) * m + (j0 + 1)

        # V indices for B entries: j*p + k
        v00 = j0 * p + k0
        v01 = j0 * p + (k0 + 1)
        v10 = (j0 + 1) * p + k0
        v11 = (j0 + 1) * p + (k0 + 1)

        # W indices for C entries: k*n + i
        w00 = k0 * n + i0
        w01 = k0 * n + (i0 + 1)
        w10 = (k0 + 1) * n + i0
        w11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (A00+A11) * (B00+B11)
        add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)],
                 [(w00, 1), (w01, 1), (w10, 1), (w11, 1)])

        # M2 = (A10+A11) * B00
        add_term([(u10, 1), (u11, 1)], [(v00, 1)],
                 [(w10, 1), (w11, 1)])

        # M3 = A00 * (B01-B11)
        add_term([(u00, 1)], [(v01, 1), (v11, -1)],
                 [(w01, 1), (w11, 1)])

        # M4 = A11 * (B10-B00)
        add_term([(u11, 1)], [(v10, 1), (v00, -1)],
                 [(w10, 1), (w11, -1)])

        # M5 = (A00+A10) * B11
        add_term([(u00, 1), (u10, 1)], [(v11, 1)],
                 [(w00, -1), (w01, -1), (w10, -1), (w11, 1)])

        # M6 = (A11-A10) * (B00+B10)
        add_term([(u11, 1), (u10, -1)], [(v00, 1), (v10, 1)],
                 [(w11, 1)])

        # M7 = (A01-A11) * (B01+B11)
        add_term([(u01, 1), (u11, -1)], [(v01, 1), (v11, 1)],
                 [(w00, 1), (w01, 1)])

    # Apply Strassen for all blocks: i0=0, j0 in {0,2}, k0 in {0,2,4,6}
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                strassen_block(i0, j0, k0)

    # Now we need to convert from sparse lists to dense matrices
    # But the interface expects {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # The entries should be integer/half-integer
    
    # U has n*m rows, rank columns
    # V has m*p rows, rank columns
    # W has p*n rows, rank columns
    
    U_dense = [[0] * rank for _ in range(n * m)]
    V_dense = [[0] * rank for _ in range(m * p)]
    W_dense = [[0] * rank for _ in range(p * n)]

    for col in range(rank):
        for (idx, val) in U[col]:
            U_dense[idx][col] = val
        for (idx, val) in V[col]:
            V_dense[idx][col] = val
        for (idx, val) in W[col]:
            W_dense[idx][col] = val

    return {"U": U_dense, "V": V_dense, "W": W_dense}
# EVOLVE-BLOCK-END
