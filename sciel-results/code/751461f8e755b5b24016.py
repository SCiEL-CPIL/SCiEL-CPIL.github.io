import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140

    terms = []

    def make_term(u_sparse, v_sparse, w_sparse):
        u = [0] * (n * m)
        for idx, val in u_sparse:
            u[idx] += val
        v = [0] * (m * p)
        for idx, val in v_sparse:
            v[idx] += val
        w = [0] * (p * n)
        for idx, val in w_sparse:
            w[idx] += val
        return (u, v, w)

    def strassen_2x2(i_off, j_off, k_off):
        """
        Generate 7 Strassen terms for a 2x2*2x2 block multiplication.
        i_off, j_off, k_off are the top-left offsets in the full matrices.
        The 2x2 blocks span:
          A: rows [i_off, i_off+1], cols [j_off, j_off+1]
          B: rows [j_off, j_off+1], cols [k_off, k_off+1]
          C: rows [i_off, i_off+1], cols [k_off, k_off+1]
        """
        result = []

        # M1 = (A11+A22)*(B11+B22)
        u = [(i_off * m + j_off, 1), ((i_off+1) * m + (j_off+1), 1)]
        v = [(j_off * p + k_off, 1), ((j_off+1) * p + (k_off+1), 1)]
        w = [(k_off * n + i_off, 1), ((k_off+1) * n + (i_off+1), 1)]
        result.append(make_term(u, v, w))

        # M2 = (A21+A22)*B11
        u = [((i_off+1) * m + j_off, 1), ((i_off+1) * m + (j_off+1), 1)]
        v = [(j_off * p + k_off, 1)]
        w = [(k_off * n + (i_off+1), 1), ((k_off+1) * n + (i_off+1), -1)]
        result.append(make_term(u, v, w))

        # M3 = A11*(B12-B22)
        u = [(i_off * m + j_off, 1)]
        v = [(j_off * p + (k_off+1), 1), ((j_off+1) * p + (k_off+1), -1)]
        w = [((k_off+1) * n + i_off, 1), ((k_off+1) * n + (i_off+1), 1)]
        result.append(make_term(u, v, w))

        # M4 = A22*(B21-B11)
        u = [((i_off+1) * m + (j_off+1), 1)]
        v = [((j_off+1) * p + k_off, 1), (j_off * p + k_off, -1)]
        w = [(k_off * n + i_off, 1), (k_off * n + (i_off+1), 1)]
        result.append(make_term(u, v, w))

        # M5 = (A11+A12)*B22
        u = [(i_off * m + j_off, 1), (i_off * m + (j_off+1), 1)]
        v = [((j_off+1) * p + (k_off+1), 1)]
        w = [((k_off+1) * n + i_off, 1), (k_off * n + i_off, -1)]
        result.append(make_term(u, v, w))

        # M6 = (A21-A11)*(B11+B12)
        u = [((i_off+1) * m + j_off, 1), (i_off * m + j_off, -1)]
        v = [(j_off * p + k_off, 1), (j_off * p + (k_off+1), 1)]
        w = [((k_off+1) * n + (i_off+1), 1)]
        result.append(make_term(u, v, w))

        # M7 = (A12-A22)*(B21+B22)
        u = [(i_off * m + (j_off+1), 1), ((i_off+1) * m + (j_off+1), -1)]
        v = [((j_off+1) * p + k_off, 1), ((j_off+1) * p + (k_off+1), 1)]
        w = [(k_off * n + i_off, 1)]
        result.append(make_term(u, v, w))

        return result

    # Recursive Strassen for 4x4*4x4
    # The 4x4 blocks are:
    # A11: i_off=0, j_off=0
    # A12: i_off=0, j_off=2
    # A21: i_off=2, j_off=0
    # A22: i_off=2, j_off=2
    # B11: j_off=0, k_off=0
    # B12: j_off=0, k_off=2
    # B21: j_off=2, k_off=0
    # B22: j_off=2, k_off=2

    # For the recursive Strassen, each of the 7 products is a 2x2*2x2 where
    # the "2x2" matrices are formed by taking one element from each of the
    # four 2x2 blocks.

    def strassen_virtual(a_positions, b_positions, c_positions):
        """
        Generate 7 Strassen terms for a virtual 2x2*2x2 multiplication.
        a_positions: list of 4 (i,j) pairs for A11, A12, A21, A22
        b_positions: list of 4 (j,k) pairs for B11, B12, B21, B22
        c_positions: list of 4 (k,i) pairs for C11, C12, C21, C22
        """
        result = []
        A11 = a_positions[0]
        A12 = a_positions[1]
        A21 = a_positions[2]
        A22 = a_positions[3]

        B11 = b_positions[0]
        B12 = b_positions[1]
        B21 = b_positions[2]
        B22 = b_positions[3]

        C11 = c_positions[0]
        C12 = c_positions[1]
        C21 = c_positions[2]
        C22 = c_positions[3]

        # M1 = (A11+A22)*(B11+B22)
        u = [(A11[0] * m + A11[1], 1), (A22[0] * m + A22[1], 1)]
        v = [(B11[0] * p + B11[1], 1), (B22[0] * p + B22[1], 1)]
        w = [(C11[0] * n + C11[1], 1), (C22[0] * n + C22[1], 1)]
        result.append(make_term(u, v, w))

        # M2 = (A21+A22)*B11
        u = [(A21[0] * m + A21[1], 1), (A22[0] * m + A22[1], 1)]
        v = [(B11[0] * p + B11[1], 1)]
        w = [(C21[0] * n + C21[1], 1), (C22[0] * n + C22[1], -1)]
        result.append(make_term(u, v, w))

        # M3 = A11*(B12-B22)
        u = [(A11[0] * m + A11[1], 1)]
        v = [(B12[0] * p + B12[1], 1), (B22[0] * p + B22[1], -1)]
        w = [(C12[0] * n + C12[1], 1), (C22[0] * n + C22[1], 1)]
        result.append(make_term(u, v, w))

        # M4 = A22*(B21-B11)
        u = [(A22[0] * m + A22[1], 1)]
        v = [(B21[0] * p + B21[1], 1), (B11[0] * p + B11[1], -1)]
        w = [(C11[0] * n + C11[1], 1), (C21[0] * n + C21[1], 1)]
        result.append(make_term(u, v, w))

        # M5 = (A11+A12)*B22
        u = [(A11[0] * m + A11[1], 1), (A12[0] * m + A12[1], 1)]
        v = [(B22[0] * p + B22[1], 1)]
        w = [(C12[0] * n + C12[1], 1), (C11[0] * n + C11[1], -1)]
        result.append(make_term(u, v, w))

        # M6 = (A21-A11)*(B11+B12)
        u = [(A21[0] * m + A21[1], 1), (A11[0] * m + A11[1], -1)]
        v = [(B11[0] * p + B11[1], 1), (B12[0] * p + B12[1], 1)]
        w = [(C22[0] * n + C22[1], 1)]
        result.append(make_term(u, v, w))

        # M7 = (A12-A22)*(B21+B22)
        u = [(A12[0] * m + A12[1], 1), (A22[0] * m + A22[1], -1)]
        v = [(B21[0] * p + B21[1], 1), (B22[0] * p + B22[1], 1)]
        w = [(C11[0] * n + C11[1], 1)]
        result.append(make_term(u, v, w))

        return result

    # For 4x4*4x4, the block structure is:
    # A blocks (each 2x2):
    #   A11: rows 0-1, cols 0-1 -> elements at (0,0),(0,1),(1,0),(1,1)
    #   A12: rows 0-1, cols 2-3 -> elements at (0,2),(0,3),(1,2),(1,3)
    #   A21: rows 2-3, cols 0-1 -> elements at (2,0),(2,1),(3,0),(3,1)
    #   A22: rows 2-3, cols 2-3 -> elements at (2,2),(2,3),(3,2),(3,3)
    #
    # B blocks (each 2x2):
    #   B11: rows 0-1, cols 0-1 -> elements at (0,0),(0,1),(1,0),(1,1)
    #   B12: rows 0-1, cols 2-3 -> elements at (0,2),(0,3),(1,2),(1,3)
    #   B21: rows 2-3, cols 0-1 -> elements at (2,0),(2,1),(3,0),(3,1)
    #   B22: rows 2-3, cols 2-3 -> elements at (2,2),(2,3),(3,2),(3,3)
    #
    # C blocks (each 2x2):
    #   C11: rows 0-1, cols 0-1 -> elements at (0,0),(0,1),(1,0),(1,1)
    #   C12: rows 0-1, cols 2-3 -> elements at (0,2),(0,3),(1,2),(1,3)
    #   C21: rows 2-3, cols 0-1 -> elements at (2,0),(2,1),(3,0),(3,1)
    #   C22: rows 2-3, cols 2-3 -> elements at (2,2),(2,3),(3,2),(3,3)

    # For M1 = (A11+A22)*(B11+B22):
    # Virtual A'[r,c] = A[2r, 2c] + A[2r+2, 2c+2]
    # Virtual B'[r,c] = B[2r, 2c] + B[2r+2, 2c+2]
    # Virtual C'[r,c] = C[2r, 2c]

    # M1 = (A11+A22)*(B11+B22)
    a_positions = [
        [(0, 0), (2, 2)],  # A11
        [(0, 1), (2, 3)],  # A12
        [(1, 0), (3, 2)],  # A21
        [(1, 1), (3, 3)]   # A22
    ]
    b_positions = [
        [(0, 0), (2, 2)],  # B11
        [(0, 1), (2, 3)],  # B12
        [(1, 0), (3, 2)],  # B21
        [(1, 1), (3, 3)]   # B22
    ]
    c_positions = [
        [(0, 0)],  # C11
        [(0, 1)],  # C12
        [(1, 0)],  # C21
        [(1, 1)]   # C22
    ]
    terms.extend(strassen_virtual(a_positions, b_positions, c_positions))

    # M2 = (A21+A22)*B11
    a_positions = [
        [(0, 0)],  # A11
        [(0, 1)],  # A12
        [(1, 0), (3, 2)],  # A21
        [(1, 1), (3, 3)]   # A22
    ]
    b_positions = [
        [(0, 0)],  # B11
        [(0, 1)],  # B12
        [(1, 0)],  # B21
        [(1, 1)]   # B22
    ]
    c_positions = [
        [(0, 0)],  # C11
        [(0, 1)],  # C12
        [(1, 0)],  # C21
        [(1, 1)]   # C22
    ]
    terms.extend(strassen_virtual(a_positions, b_positions, c_positions))

    # M3 = A11*(B12-B22)
    a_positions = [
        [(0, 0)],  # A11
        [(0, 1)],  # A12
        [(1, 0)],  # A21
        [(1, 1)]   # A22
    ]
    b_positions = [
        [(0, 0)],  # B11
        [(0, 1)],  # B12
        [(1, 0)],  # B21
        [(1, 1)]   # B22
    ]
    c_positions = [
        [(0, 0)],  # C11
        [(0, 1)],  # C12
        [(1, 0)],  # C21
        [(1, 1)]   # C22
    ]
    terms.extend(strassen_virtual(a_positions, b_positions, c_positions))

    # M4 = A22*(B21-B11)
    a_positions = [
        [(0, 0)],  # A11
        [(0, 1)],  # A12
        [(1, 0)],  # A21
        [(1, 1), (3, 3)]   # A22
    ]
    b_positions = [
        [(0, 0)],  # B11
        [(0, 1)],  # B12
        [(1, 0)],  # B21
        [(1, 1)]   # B22
    ]
    c_positions = [
        [(0, 0)],  # C11
        [(0, 1)],  # C12
        [(1, 0)],  # C21
        [(1, 1)]   # C22
    ]
    terms.extend(strassen_virtual(a_positions, b_positions, c_positions))

    # M5 = (A11+A12)*B22
    a_positions = [
        [(0, 0)],  # A11
        [(0, 1)],  # A12
        [(1, 0)],  # A21
        [(1, 1)]   # A22
    ]
    b_positions = [
        [(0, 0)],  # B11
        [(0, 1)],  # B12
        [(1, 0)],  # B21
        [(1, 1), (3, 3)]   # B22
    ]
    c_positions = [
        [(0, 0)],  # C11
        [(0, 1)],  # C12
        [(1, 0)],  # C21
        [(1, 1)]   # C22
    ]
    terms.extend(strassen_virtual(a_positions, b_positions, c_positions))

    # M6 = (A21-A11)*(B11+B12)
    a_positions = [
        [(0, 0)],  # A11
        [(0, 1)],  # A12
        [(1, 0), (3, 2)],  # A21
        [(1, 1)]   # A22
    ]
    b_positions = [
        [(0, 0)],  # B11
        [(0, 1)],  # B12
        [(1, 0)],  # B21
        [(1, 1)]   # B22
    ]
    c_positions = [
        [(0, 0)],  # C11
        [(0, 1)],  # C12
        [(1, 0)],  # C21
        [(1, 1)]   # C22
    ]
    terms.extend(strassen_virtual(a_positions, b_positions, c_positions))

    # M7 = (A12-A22)*(B21+B22)
    a_positions = [
        [(0, 0)],  # A11
        [(0, 1)],  # A12
        [(1, 0)],  # A21
        [(1, 1), (3, 3)]   # A22
    ]
    b_positions = [
        [(0, 0)],  # B11
        [(0, 1)],  # B12
        [(1, 0)],  # B21
        [(1, 1)]   # B22
    ]
    c_positions = [
        [(0, 0)],  # C11
        [(0, 1)],  # C12
        [(1, 0)],  # C21
        [(1, 1)]   # C22
    ]
    terms.extend(strassen_virtual(a_positions, b_positions, c_positions))

    # Schoolbook for 4x4 * 4x3 (k=4,5,6)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = [(i * m + j, 1)]
                v = [(j * p + k, 1)]
                w = [(k * n + i, 1)]
                terms.append(make_term(u, v, w))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
